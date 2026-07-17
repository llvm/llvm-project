==========================================
WebAssembly Function Pointer Bitcast Fix
==========================================

.. contents::
   :local:

Overview
========

WebAssembly enforces strict function signature matching at ``call_indirect``
sites. Native platforms silently allow calling a function through a pointer
with a different signature (e.g., passing a 2-parameter function where a
3-parameter function is expected). On WebAssembly, this causes a
``RuntimeError: function signature mismatch``.

The ``-fwasm-fix-function-bitcasts`` flag generates adapter thunks to bridge
incompatible signatures at function pointer cast sites.

**Constraints**:

- ``SrcParams > DstParams`` → rejected (can't remove parameters — can't
  invent values the caller didn't provide)
- ``DstReturn != void && DstReturn != SrcReturn`` → rejected (can't invent
  return values)
- Identical signatures → skipped (no adaptation needed)

1. Static Casts
===============

**What they are**: A cast from one function pointer type to another where the
compiler can statically trace the original function. The function name is a
compile-time constant.

**C code example**:

.. code-block:: c

    typedef void (*OneArg)(void*);
    typedef void (*TwoArg)(void*, void*);

    void my_func(void *p) { }

    void caller() {
        TwoArg f = (TwoArg)my_func;   // my_func is known at compile time
        f(data, NULL);
    }

**Solution**: One thunk function per (``original_function``,
``dest_signature``) pair, cached by name. The thunk receives dest params, calls
the original with src params.

**Generated code**:

.. code-block:: llvm

    ; Thunk: __my_func_vii
    define internal void @__my_func_vii(ptr %0, ptr %1) {
        call void @my_func(ptr %0)    ; drops 2nd param
        ret void
    }

**Cache**: ``ThunkCache`` — ``StringMap<Function*>`` keyed by
``__<name>_<dest_sig>``.

2. Runtime Casts
================

**What they are**: A cast where the source function pointer is a runtime value
(function parameter, loaded from memory, etc.), not a compile-time constant.

**C code example**:

.. code-block:: c

    void caller(OneArg fp) {
        TwoArg f = (TwoArg)fp;        // fp is unknown at compile time
        f(data, NULL);
    }

Runtime casts are split into two sub-cases based on usage pattern:

2.1 Immediate Calls
-------------------

**What they are**: The cast result is immediately called (the cast expression is
the callee of a ``CallExpr``). The wrapper is used and discarded within the same
expression — no other code can observe the adapted pointer.

**Detection**: The cast ``Expr``'s parent is a ``CallExpr`` and the cast is the
callee.

**C code example**:

.. code-block:: c

    void caller(OneArg fp, void *data) {
        ((TwoArg)fp)(data, NULL);     // cast + immediate call
    }

**Solution**: One TLS slot + one wrapper function per signature pair per
translation unit. The slot is thread-local — no races across threads.

**Generated code**:

.. code-block:: llvm

    ; One TLS slot per signature pair per TU:
    @__wasm_runtime_pool___wasm_runtime_wrapper_iii_to_iiii____source_c_immediate_slot
        = internal thread_local global ptr null

    ; One wrapper, loads from TLS slot:
    define internal void @__wasm_runtime_wrapper_iii_to_iiii____source_c_immediate(ptr %0, ptr %1) {
        %fp = load ptr, ptr @__wasm_runtime_pool___wasm_runtime_wrapper_iii_to_iiii____source_c_immediate_slot
        br %fp != null → call_bb, null_bb
      call_bb:
        call i32 %fp(ptr %0, ptr %1)   ; call source with adapted args
        unreachable                     ; dest returns void, discard result
      null_bb:
        ret void                        ; defensive null return
    }

**Runtime (at each cast site)**:

.. code-block:: llvm

    br %fn_ptr == null → null_cont, not_null

  not_null:
    store ptr %fn_ptr, ptr @__wasm_runtime_pool___wasm_runtime_wrapper_iii_to_iiii____source_c_immediate_slot
    br null_cont

  null_cont:
    %w = phi [ @__wasm_runtime_wrapper_iii_to_iiii____source_c_immediate, not_null ],
             [ null, null_cont ]

**Characteristics**:

- 1 TLS slot, 1 wrapper per signature pair per TU
- No counter, no atomics
- Thread-safe via TLS
- Reused every call — never runs out

2.2 Store-For-Later (Closures)
------------------------------

**What they are**: The cast result is assigned to a variable, stored in a struct
field, or passed to another function — the adapted pointer persists beyond the
current expression. Multiple such stores may be active simultaneously.

**Detection**: The cast ``Expr`` is NOT the callee of a ``CallExpr`` — this
includes assignments, function arguments, initializers, struct field stores, and
return statements.

**C code example**:

.. code-block:: c

    // Closure pattern: multiple marshals stored in struct field
    typedef void (*GMarshal)(void*, void*, void*, void*, void*, void*);

    struct Closure {
        GMarshal marshal;   // stored for later invocation
    };

    void set_closure(Closure *c, void (*notify)(void*, void*)) {
        c->marshal = (GMarshal)notify;    // cast + store in struct
    }

**Corner cases — syntactically "store" but semantically "immediate"**:

Some patterns look like "store-for-later" at the syntax level but are actually
immediate calls from a single-threaded perspective. These are treated as
store-for-later for safety:

1. **Assign to local variable, then call**:

   .. code-block:: c

       TwoArg f = (TwoArg)fp;   // assignment → classified as store-for-later
       f(a, b);                 // immediately called, but we can't track this

   The cast is in an assignment, not a call callee. Our AST-level detection
   sees this as a store. We conservatively treat it as store-for-later.

2. **Pass cast result as function argument**:

   .. code-block:: c

       other_func((TwoArg)fp);  // passed as parameter → store-for-later

   The cast result is passed to another function. We cannot know what
   ``other_func`` does with it — it might call it immediately, store it in a
   global, or pass it to another thread. We conservatively use the pool.

**Solution**: Pre-allocated pool of 64 wrappers + 64 non-TLS slots + 8-entry
direct-mapped cache + atomic counter per signature pair per translation unit.

**Generated code (pool setup, once per TU per signature pair)**:

.. code-block:: llvm

    ; 64 wrapper functions, each loading from its dedicated slot:
    ; Wrapper 0:
    define internal void @__wasm_runtime_wrapper_vii_to_viiiiii____gobject_gclosure_c_0(
        ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5) {
        %fp = load ptr, ptr getelementptr(ptr, @slots, i32 0)
        br %fp != null → call_bb, null_bb
      call_bb:
        call void %fp(ptr %0, ptr %1)    ; source: 2 params
        ret void
      null_bb:
        ret void
    }
    ; Wrappers 1-63: same pattern, each loads slots[1] through slots[63]

    ; Pool globals:
    @counter  = internal thread_local global i32 0
    @slots    = internal global [64 x ptr] zeroinitializer
    @wrappers = internal constant [64 x ptr] [ @wrapper_0, ..., @wrapper_63 ]

    ; 8-entry direct-mapped cache:
    @cache_keys     = internal global [8 x ptr] zeroinitializer
    @cache_wrappers = internal global [8 x ptr] zeroinitializer

**Runtime flow (at each cast site)**:

1. **Cache lookup** — O(1): hash fn_ptr, check cache. Hit → return cached wrapper.
2. **Pool scan** — O(64): linear scan of slots. Found → update cache, return wrapper.
3. **Allocate** — O(1): atomic increment counter, check overflow, store fn_ptr + update cache.

Overflow (>64 unique fn_ptrs) traps via ``llvm.trap``.

**Deduplication**: Cache + pool scan ensures each fn_ptr maps to exactly one
wrapper, even across cache evictions. Same fn_ptr always returns same wrapper.

**Characteristics**:

- 64 wrappers + 64 non-TLS slots + 8-entry cache per signature pair per TU
- Deterministic: same fn_ptr always returns same wrapper (no duplicate allocation)
- O(1) cache hit, O(64) scan on first miss, O(1) atomic on allocation
- Atomic counter (TLS), shared slots + cache (non-TLS)
- Traps if >64 **unique** fn_ptrs cast in store-for-later context
- Each wrapper has a null check for defensive safety

3. Non-Supported Cases and Limitations
======================================

3.1 Standard Compliance
-----------------------

The C and C++ standards specify exactly one guarantee for function pointer
casts:

    Converting a prvalue of type "pointer to ``T1``" to the type "pointer to
    ``T2``" (where ``T1`` and ``T2`` are function types) and back to its
    original type yields the original pointer value.

Our adapter thunks violate this guarantee. Each cast returns a **different**
pointer — the wrapper function's address:

.. code-block:: c

    typedef void (*OneArg)(void*);
    typedef void (*TwoArg)(void*, void*);

    OneArg f = my_func;
    TwoArg g = (TwoArg)f;     // returns wrapper function pointer, NOT f
    OneArg h = (OneArg)g;     // returns another wrapper, NOT f
    // h != f  ← violates round-trip guarantee

This is an accepted deviation: WebAssembly's type system cannot express the
standard's semantics. The alternative is a runtime crash on ``call_indirect``.

3.2 Chained Casts Through Memory
--------------------------------

When a function pointer is cast, stored in a struct field, loaded back, and
cast again, the chain of casts cannot be tracked at compile time. Each cast
sees only the **declared** type of the source expression, not the **actual**
type of the function that was originally stored.

**Real-world example — gstreamer ``gstutils.c``**:

.. code-block:: c

    // Type definitions:
    typedef void (*GFunc)(gpointer, gpointer);              // vii
    typedef void (*GstCallAsyncFunc)(gpointer);              // vi
    typedef void (*GstObjectCallAsyncFunc)(GstObject*, gpointer); // vii

    struct GstCallAsyncData {
        GstObject *object;
        GstCallAsyncFunc func;   // declared as vi (1 param)
        gpointer user_data;
    };

    // Store: vii → vi → REJECTED (can't remove params) → raw bitcast
    data->func = (GstCallAsyncFunc)vii_func;

    // Load: vi → vii → looks valid, wraps. But stored func is actually vii.
    // Wrapper calls it as vi (1 param), but it's vii (2 params) → MISMATCH.
    GstObjectCallAsyncFunc func = (GstObjectCallAsyncFunc)data->func;
    (*func)(data->object, data->user_data);

**Why it fails**: The struct field is declared as ``vi`` but sometimes stores a
``vii`` function. Our code sees ``vi → vii`` at the load site (valid) but the
actual function in memory is ``vii`` (from a rejected store). This is a
pre-existing source bug — the struct field type doesn't match what's actually
stored there.

**General rule**: When a struct field holds multiple function types through
different code paths, our per-cast-site adaptation cannot determine the correct
runtime type. The fix belongs in the source code (use ``gpointer`` for the
field, or a union, or ensure consistent types).

3.3 Other Limitations
---------------------

- **Source params > Dest params**: Cannot remove parameters — we can't invent
  values the caller didn't provide. Falls through to raw bitcast (wasm crash).
- **Dest returns non-void and doesn't match source return**: Cannot invent
  return values. Falls through to raw bitcast (wasm crash).
- **Identical signatures**: No adaptation needed. Skipped.
- **Null function pointers**: Skipped — falls through to raw bitcast (wasm
  would crash on null call regardless).

Key Source Files
================

- ``clang/lib/CodeGen/Targets/WebAssembly.cpp`` — thunk/wrapper generation
- ``clang/lib/CodeGen/CGExprScalar.cpp`` — cast site detection and dispatch
- ``clang/lib/CodeGen/TargetInfo.h`` — virtual interface
- ``clang/test/CodeGenWebAssembly/function-pointer-runtime-cast.c`` — LLVM IR tests
