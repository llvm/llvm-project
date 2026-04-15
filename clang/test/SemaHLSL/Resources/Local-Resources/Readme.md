# Local Resource Tests

## Purpose

This test suite documents and verifies the behavior of **local resource variables** in HLSL — that is, resource handles declared within function scope rather than at global scope. Local resources are a common pattern in HLSL shaders, yet their semantics around initialization, assignment, aliasing, and control flow have historically been under-documented and inconsistently handled across compilers.

## Motivation

DXC has never had structured test coverage for local resource patterns. Many valid and invalid usage patterns were either untested or bundled into monolithic test files, making it difficult to understand what is intentional behavior versus incidental. This suite was created to:

1. **Establish a reference** for how both Clang and DXC handle each local resource pattern.
2. **Document behavioral differences** between the two compilers, particularly where DXC issues hard errors for patterns that Clang treats as warnings or accepts silently.
3. **Enable regression testing** so that future compiler changes can be validated against well-defined expectations.
4. **Provide individual, self-contained tests** — each file tests exactly one pattern with its own globals, helpers, and entry point, making failures easy to diagnose.

## Organization

Tests are split across multiple directories based on compiler behavior:

| Location | Count | Contents |
|----------|-------|----------|
| `SemaHLSL/Resources/Local-Resources/` (this directory) | 15 | Fail to compile on **both** compilers (invalid type ops, bad declarations) |
| `CodeGenHLSL/resources/Local-Resources/` | 11 | Codegen-stage tests (DXC codegen failures, groupshared) |
| `offload-test-suite/test/Feature/LocalResources/` | 55 | Produce compiled output on **both** compilers (clean, or Clang warns) |
| `offload-test-suite/test/Feature/LocalResources/ClangPass/` | 4 | Clang compiles, DXC fails (ICE or codegen error) |
| `offload-test-suite/test/Feature/LocalResources/DXCPass/` | 1 | DXC compiles, Clang fails to compile |
| **Total** | **86** | |

### `SemaHLSL/Resources/Local-Resources/` (this directory)

Tests where **sema is the interesting stage** and **both compilers reject the code**. These are expected errors caught by `-verify`.

### `CodeGenHLSL/resources/Local-Resources/`

Tests where **codegen is the interesting stage**. This includes:
- Tests that **pass sema but fail codegen** in either compiler. These use `FileCheck` to verify that Clang produces valid IR and emits the expected diagnostics, while documenting DXC's codegen-stage errors in comments.

### `offload-test-suite/test/Feature/LocalResources/`

Tests where **both compilers produce a compiled output**. This includes tests that are fully clean on both compilers, as well as tests where Clang emits `-Whlsl-explicit-binding` warnings but still compiles successfully. The `ClangPass/` and `DXCPass/` subdirectories contain tests that compile on only one compiler.

## Test Categories

### Basic Local Resource Operations
| File | Pattern |
|------|---------|
| `local_resource_alias_global.hlsl` | Local resource initialized from a global |
| `local_resource_alias_chain.hlsl` | Chain of local-to-local aliases (`a = g; b = a; c = b`) |
| `local_resource_copy_between_locals.hlsl` | Copy one local resource to another (`a = g; b = a`) |
| `local_resource_self_assign.hlsl` | Self-assignment (`buf = buf`) |
| `use_local_resource_uninitialized.hlsl` | Using an uninitialized local resource |
| `local_resource_default_init_store.hlsl` | Store through a default-initialized (unbound) resource *(both compilers pass)* |
| `return_local_resource_uninitialized.hlsl` | Returning an uninitialized local resource |
| `return_local_resource_initialized.hlsl` | Returning a local resource initialized from a global |
| `expression_init.hlsl` | Initializing with a parenthesized ternary expression |
| `local_resource_aggregate_init.hlsl` | Aggregate initialization of a struct containing a resource |
| `local_resource_multi_decl.hlsl` | Two resource variables declared in a single statement |
| `local_resource_comma_init.hlsl` | Comma expression in initializer — left operand discarded with warning *(clang warns, DXC silent)* |

### Parameter Passing
| File | Pattern |
|------|---------|
| `local_resource_out_param.hlsl` | Resource written through an `out` parameter |
| `local_resource_inout_param.hlsl` | Resource passed as an `inout` parameter |
| `local_resource_const_param.hlsl` | Resource passed as a `const` parameter |

### Struct and Array Patterns
| File | Pattern |
|------|---------|
| `local_struct_resource_member.hlsl` | Struct with a resource member |
| `struct_array_with_resource_member.hlsl` | Array of structs, each containing a resource |
| `struct_with_resource_array_member.hlsl` | Struct containing an array of resources |
| `nested_struct_resource_member.hlsl` | Inner/outer nested struct with resource |
| `forward_struct_layers_resource.hlsl` | Deeply composed struct layers with resource |
| `local_resource_array.hlsl` | Plain local array of resources |
| `local_resource_array_copy.hlsl` | Copy one local resource array to another |
| `local_resource_array_dynamic_index.hlsl` | Runtime dynamic index into a local resource array |
| `local_resource_array_partial_init.hlsl` | Partially initialized resource array (`{g0}` for a 2-element array) |
| `local_resource_array_size_one.hlsl` | Local resource array of size 1 (edge case) |
| `struct_resource_member_reassign.hlsl` | Reassign a struct's resource member to a different global |
| `local_resource_struct_return.hlsl` | Function returns a struct containing a resource |
| `local_resource_mixed_struct.hlsl` | Struct with both a resource member and a scalar member |
| `local_resource_struct_method.hlsl` | User-defined struct with a member function that uses a resource |

### Control Flow
| File | Pattern |
|------|---------|
| `local_resource_shadow_inner_scope.hlsl` | Shadowed resource in an inner block |
| `local_resource_block_lifetime.hlsl` | Assigned in inner block, used in outer scope |
| `local_resource_conditional_init.hlsl` | Resource only initialized in one branch of an if *(both compilers pass)* |
| `local_resource_nested_blocks_reassign.hlsl` | Reassigned across nested blocks *(clang warns, DXC silent)* |
| `local_resource_early_return_reassign.hlsl` | Reassigned after an early return path *(clang warns, DXC silent)* |
| `local_resource_unreachable_reassign.hlsl` | Reassigned in code after an early return *(clang warns, DXC silent)* |
| `local_resource_switch_reassign.hlsl` | Reassigned in switch cases *(clang warns, DXC silent)* |
| `local_resource_switch_fallthrough.hlsl` | Switch with fallthrough reassignment *(clang warns, DXC silent)* |
| `local_resource_switch_default.hlsl` | Switch with explicit default case reassignment *(clang warns, DXC silent)* |

### Loop Patterns
| File | Pattern |
|------|---------|
| `loop_var.hlsl` | Resource as a for-loop variable |
| `local_resource_loop_array_index.hlsl` | Resource from array inside a loop |
| `local_resource_nested_loops.hlsl` | Resource from array in nested loops |
| `local_resource_loop_carried.hlsl` | Loop-carried reassignment from array *(clang warns, DXC silent)* |
| `local_resource_do_while_reassign.hlsl` | Reassignment inside a do-while loop *(clang warns, DXC silent)* |

### Reassignment and Phi/Merge
| File | Pattern |
|------|---------|
| `local_resource_reassign_different_global.hlsl` | Reassign to a different global *(clang warns, DXC silent)* |
| `local_resource_deep_phi.hlsl` | Nested if/else with ternary *(clang warns, DXC silent)* |
| `local_resource_ternary_lvalue.hlsl` | Ternary expression as lvalue for resource assignment *(clang silent, DXC ICEs)* |
| `local_resource_swap.hlsl` | Swap two local resources through a temporary *(both compilers pass, no warnings)* |

### Bindless
| File | Pattern |
|------|---------|
| `local_resource_bindless_array.hlsl` | Dynamic index into global resource array |
| `local_resource_bindless_selection.hlsl` | Multiple dynamic array selections |

### Function Forwarding and Multiple Uses
| File | Pattern |
|------|---------|
| `local_resource_forward_through_functions.hlsl` | Resource passed through a call chain |
| `local_resource_with_wave_intrinsic.hlsl` | Resource used alongside wave intrinsics |
| `local_resource_multiple_uses.hlsl` | Same local resource passed to multiple helper functions |
| `local_resource_from_function_return.hlsl` | Local resource initialized from a function's return value |
| `local_resource_template_function.hlsl` | Template function taking a resource parameter |
| `local_resource_chained_call.hlsl` | Method called directly on a function return value (`GetBuf().Store(...)`) |
| `local_resource_overload.hlsl` | Function overloading where overloads differ by resource type |

### Static and Storage
| File | Pattern |
|------|---------|
| `local_resource_static_local.hlsl` | Static local resource initialized from a global *(passes both compilers with RWByteAddressBuffer; DXC ICEs with Texture2D)* |

### Type Mixing and Alternative Resource Types
| File | Pattern |
|------|---------|
| `local_resource_different_types.hlsl` | Two different resource types (`RWByteAddressBuffer` + `RWStructuredBuffer`) in the same function |
| `local_resource_read_only.hlsl` | Read-only `ByteAddressBuffer` as local — only `Load` available (no `Store`) |
| `local_resource_structured_buffer.hlsl` | `RWStructuredBuffer<uint>` as local with subscript operator access |

### Invalid Type Operations (Sema Failures)
| File | Pattern |
|------|---------|
| `local_resource_arithmetic.hlsl` | Arithmetic (`buf + 1`) on a resource handle |
| `local_resource_addition.hlsl` | Addition (`buf + buf`) of two resource handles |
| `local_resource_compare.hlsl` | Equality comparison (`a == b`) of two resources |
| `local_resource_to_bool.hlsl` | Implicit conversion of a resource to `bool` |
| `local_resource_cast_to_uint.hlsl` | C-style cast of a resource to `uint` |
| `local_resource_cast_sampler_to_buffer.hlsl` | C-style cast from `SamplerState` to `RWByteAddressBuffer` |
| `local_resource_assign_wrong_type.hlsl` | Assign `RWStructuredBuffer` to `RWByteAddressBuffer` (type mismatch) |
| `local_resource_volatile.hlsl` | `volatile` qualifier on resource prevents method calls *(clang errors, DXC accepts)* |
| `local_resource_const_reassign.hlsl` | `const` local resource prevents reassignment *(both compilers error)* |
| `local_resource_static_const.hlsl` | `static const` prevents calling any methods — `Load` and `Store` are not `const`-qualified *(clang errors, DXC ICEs)* |

### Invalid Declarations (Sema Failures)
| File | Pattern |
|------|---------|
| `local_resource_default_param.hlsl` | Resource parameter with default followed by parameter without default |
| `local_resource_as_structured_buffer_element.hlsl` | Resource type as element of `RWStructuredBuffer` (intangible type) |
| `local_resource_array_oob.hlsl` | Compile-time out-of-bounds index into resource array *(clang warns, DXC errors)* |
| `local_resource_zero_init.hlsl` | Brace (zero) initialization `= {}` rejected — empty initializer list *(both compilers error)* |

### Groupshared Resources (now in `CodeGenHLSL/resources/Local-Resources/`)
| File | Pattern |
|------|---------|
| `use_groupshared.hlsl` | Passing a groupshared resource as a function argument *(clang sema error; DXC validation error)* |
| `use_groupshared_direct_store.hlsl` | Calling Store directly on a groupshared resource *(fails sema — address space mismatch)* |
| `use_struct_groupshared.hlsl` | Using a resource from a groupshared struct *(expected to fail — see TODO in file; DXC validation error)* |

### CodeGen Tests (`CodeGenHLSL/resources/Local-Resources/`)
| File | Pattern |
|------|---------|
| `ternary_initialization.hlsl` | Ternary init (`buf = cond ? g0 : g1`) *(DXC codegen error)* |
| `local_resource_ternary_assign.hlsl` | Ternary assignment post-declaration *(DXC codegen error)* |
| `local_resource_phi_merge_ternary.hlsl` | Ternary phi merge *(DXC codegen error)* |
| `local_resource_wave_uniform.hlsl` | Wave-conditional reassignment *(DXC codegen error; clang warns)* |
| `local_resource_break_reassign.hlsl` | Reassignment before a break in a loop *(DXC codegen error; clang warns)* |
| `local_resource_ternary_as_argument.hlsl` | Ternary resource passed directly as a function argument *(DXC codegen error; clang silent)* |
| `local_resource_nested_ternary.hlsl` | Nested ternary (`c1 ? g0 : (c2 ? g1 : g2)`) *(DXC codegen 2 errors; clang warns)* |
| `local_resource_continue_reassign.hlsl` | Reassignment before `continue` in a loop *(DXC codegen error; clang warns)* |
| `local_resource_multiple_returns.hlsl` | Multiple return paths returning different resources *(DXC codegen error; clang warns)* |
| `use_groupshared.hlsl` | Passing a groupshared resource as a function argument *(clang sema error; DXC validation error)* |
| `use_struct_groupshared.hlsl` | Using a resource from a groupshared struct *(expected failure — clang currently silent; DXC validation error)* |

## Key Behavioral Differences: Clang vs DXC

### `-Whlsl-explicit-binding` (Clang-only)
Clang emits warnings when a local resource is assigned from a source that does not resolve to a single unique global resource. DXC has no equivalent diagnostic — it either silently accepts the pattern or rejects it as a hard error during codegen.

### Ternary Conditional Resource Assignment
DXC's `DxilCondenseResources` pass rejects `cond ? gBuf0 : gBuf1` as a codegen error: *"local resource not guaranteed to map to unique global resource."* Clang accepts this pattern and emits a warning.

### Groupshared Resources
Both compilers reject using groupshared resources in most contexts, but the specific diagnostics differ. Clang rejects at sema with constructor-mismatch errors. DXC rejects during validation.

### Sema Error Messages
For type-error tests (arithmetic, comparison, cast, etc.), both compilers reject the invalid code at sema, but the error messages often differ. Clang uses its standard C++ diagnostics (e.g., "invalid operands to binary expression") while DXC uses HLSL-specific messages (e.g., "scalar, vector, or matrix expected"). Each test file documents both.

### Texture2D vs RWByteAddressBuffer
The failure tests use `RWByteAddressBuffer` rather than `Texture2D` because Clang does not yet support `Texture2D`. Some behaviors differ between the two resource types — notably, DXC asserts/ICEs with `static Texture2D` but accepts `static RWByteAddressBuffer`. This is documented in `local_resource_static_local.hlsl`.

### Wave-Conditional Reassignment
DXC's `DxilCondenseResources` pass rejects resource reassignment under wave-conditional control flow as a hard codegen error: *"local resource not guaranteed to map to unique global resource."* DXC's sema passes this pattern silently. Clang treats it as a sema-level warning (`-Whlsl-explicit-binding`). This pattern is in `CodeGenHLSL` because of the DXC codegen failure.

### `volatile` Resources
Clang rejects calling methods on a `volatile`-qualified resource because the methods are not marked `volatile`. DXC silently accepts `volatile` on resources and compiles successfully. This is documented in `local_resource_volatile.hlsl`.

### Conditional/Partial Initialization
DXC rejects resources that are only initialized in one branch (`if(cond) buf = g;`) or that use default-initialized (unbound) handles. These patterns fail in DXC's `DxilCondenseResources` pass with *"local resource not guaranteed to map to unique global resource."* Clang accepts them silently. See `local_resource_conditional_init.hlsl` and `local_resource_default_init_store.hlsl` in `CodeGenHLSL`.

### Array Out-of-Bounds
Both compilers catch compile-time OOB indexing into resource arrays, but at different severity levels. Clang emits a warning (`-Warray-bounds`); DXC emits a hard error. See `local_resource_array_oob.hlsl`.
