// Check the "GS buffer" classification clang-cl's default /GS applies to each
// stack object. The result is recorded as "stack-protector" metadata on the
// alloca, because it depends on source-level type information that LLVM IR does
// not preserve.
//
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -O2 \
// RUN:     -stack-protector 4 -emit-llvm %s -o - | FileCheck %s
//
// The classification is only emitted for the /GS heuristic: the GCC-compatible
// levels infer it from the IR type instead. The one exception is an explicit
// opt-out, which is not a classification.
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -O2 \
// RUN:     -stack-protector 2 -emit-llvm %s -o - | FileCheck %s --check-prefix=STRONG
//
// STRONG-NOT: = !{i32 1}
// STRONG-NOT: = !{i32 2}
// STRONG-NOT: = !{i32 1, i1
// STRONG-NOT: = !{i32 2, i1

extern "C" void *_alloca(unsigned long long);

void use(void *);

//--- Arrays -------------------------------------------------------------------

// An array has to be larger than 4 bytes...
// CHECK-LABEL: @"?array_4@@YAXXZ"
// CHECK:         alloca [4 x i8], align 1{{$}}
void array_4() { char a[4]; use(a); }

// ...and have more than two elements.
// CHECK-LABEL: @"?array_2_elements@@YAXXZ"
// CHECK:         alloca [2 x i32], align 4{{$}}
void array_2_elements() { int a[2]; use(a); }

// For the type of the stack object itself those are the only two conditions, so
// unlike for a member (see struct_array_of_pointers) the element type does not
// matter. MSDN says otherwise, but cl.exe protects this.
// CHECK-LABEL: @"?array_of_pointers@@YAXXZ"
// CHECK:         alloca [8 x ptr], align 8, !stack-protector ![[LARGE:[0-9]+]]
void array_of_pointers() { void *a[8]; use(a); }

// A multidimensional array is counted as the flat array it is laid out as, so
// this has four elements rather than two.
// CHECK-LABEL: @"?array_2_by_2_pointers@@YAXXZ"
// CHECK:         alloca [2 x [2 x ptr]], align 8, !stack-protector ![[LARGE]]
void array_2_by_2_pointers() { void *a[2][2]; use(a); }

// A GS buffer of at least ssp-buffer-size bytes is laid out closest to the
// guard, so it is marked 2 rather than 1.
// CHECK-LABEL: @"?array_8_bytes@@YAXXZ"
// CHECK:         alloca [8 x i8], align 1, !stack-protector ![[LARGE]]
void array_8_bytes() { char a[8]; use(a); }

// 6 bytes and three elements: over both thresholds, but under ssp-buffer-size.
// CHECK-LABEL: @"?array_3_shorts@@YAXXZ"
// CHECK:         alloca [3 x i16], align 2, !stack-protector ![[SMALL:[0-9]+]]
void array_3_shorts() { short a[3]; use(a); }

// Two elements of two bytes each: too small either way.
// CHECK-LABEL: @"?array_2_chars_nested@@YAXXZ"
// CHECK:         alloca [2 x [2 x i8]], align 1{{$}}
void array_2_chars_nested() { char a[2][2]; use(a); }

//--- Aggregates ---------------------------------------------------------------

struct PointerFree { int a, b, c; };
struct WithPointer { void *p; int a, b; };
struct EightBytes { int a, b; };
struct ContainsBuffer { void *p; char buf[8]; };
struct NestedBuffer { void *p; ContainsBuffer c; };
struct ArrayOfPointers { void *m[4]; };
struct TwoIntArray { int m[2]; };

// More than 8 bytes with no pointers.
// CHECK-LABEL: @"?struct_pointer_free@@YAXXZ"
// CHECK:         alloca %struct.PointerFree, align 4, !stack-protector ![[LARGE]]
void struct_pointer_free() { PointerFree s; use(&s); }

// CHECK-LABEL: @"?struct_with_pointer@@YAXXZ"
// CHECK:         alloca %struct.WithPointer, align 8{{$}}
void struct_with_pointer() { WithPointer s; use(&s); }

// Exactly 8 bytes, so not larger than 8.
// CHECK-LABEL: @"?struct_8_bytes@@YAXXZ"
// CHECK:         alloca %struct.EightBytes, align 4{{$}}
void struct_8_bytes() { EightBytes s; use(&s); }

// Holds a pointer, but also contains a GS buffer.
// CHECK-LABEL: @"?struct_containing_buffer@@YAXXZ"
// CHECK:         alloca %struct.ContainsBuffer, align 8, !stack-protector ![[LARGE]]
void struct_containing_buffer() { ContainsBuffer s; use(&s); }

// CHECK-LABEL: @"?struct_nested_buffer@@YAXXZ"
// CHECK:         alloca %struct.NestedBuffer, align 8, !stack-protector ![[LARGE]]
void struct_nested_buffer() { NestedBuffer s; use(&s); }

// Below the top level the two array conditions swap over: the element count
// stops mattering and the element type starts to, which is the opposite of how
// the same two types fare as locals (see array_of_pointers and
// array_2_elements).
// CHECK-LABEL: @"?struct_array_of_pointers@@YAXXZ"
// CHECK:         alloca %struct.ArrayOfPointers, align 8{{$}}
void struct_array_of_pointers() { ArrayOfPointers s; use(&s); }

// CHECK-LABEL: @"?struct_two_int_array@@YAXXZ"
// CHECK:         alloca %struct.TwoIntArray, align 4, !stack-protector ![[LARGE]]
void struct_two_int_array() { TwoIntArray s; use(&s); }

//--- Anonymous members --------------------------------------------------------

// Each of these is 8 bytes, so only the member rule can make it a buffer, not
// the size rule for the aggregate as a whole.
struct AnonUnionBuffer { union { char b[8]; }; };
struct AnonStructBuffer { struct { char b[8]; }; };
struct NamedUnionBuffer { union U { char b[8]; } u; };
struct AnonUnionPointer { union { char b[16]; void *p; }; int n; };

// cl.exe does not look inside an anonymous struct or union for a buffer...
// CHECK-LABEL: @"?anon_union_buffer@@YAXXZ"
// CHECK:         alloca %struct.AnonUnionBuffer, align 1{{$}}
void anon_union_buffer() { AnonUnionBuffer s; use(&s); }

// CHECK-LABEL: @"?anon_struct_buffer@@YAXXZ"
// CHECK:         alloca %struct.AnonStructBuffer, align 1{{$}}
void anon_struct_buffer() { AnonStructBuffer s; use(&s); }

// ...though the same union as a named member is found.
// CHECK-LABEL: @"?named_union_buffer@@YAXXZ"
// CHECK:         alloca %struct.NamedUnionBuffer, align 1, !stack-protector ![[LARGE]]
void named_union_buffer() { NamedUnionBuffer s; use(&s); }

// A pointer inside one still disqualifies the enclosing aggregate.
// CHECK-LABEL: @"?anon_union_pointer@@YAXXZ"
// CHECK:         alloca %struct.AnonUnionPointer, align 8{{$}}
void anon_union_pointer() { AnonUnionPointer s; use(&s); }

//--- Access control -----------------------------------------------------------

class PrivateData { long long a, b; };
struct ProtectedData { protected: long long a, b; };
struct PrivateArray { private: char b[16]; };

// An aggregate only becomes a GS buffer by size if it is plain, publicly
// accessible data. MSDN does not mention this, but cl.exe applies it.
// CHECK-LABEL: @"?private_data@@YAXXZ"
// CHECK:         alloca %class.PrivateData, align 8{{$}}
void private_data() { PrivateData s; use(&s); }

// CHECK-LABEL: @"?protected_data@@YAXXZ"
// CHECK:         alloca %struct.ProtectedData, align 8{{$}}
void protected_data() { ProtectedData s; use(&s); }

// The array rule is not subject to it.
// CHECK-LABEL: @"?private_array@@YAXXZ"
// CHECK:         alloca %struct.PrivateArray, align 1, !stack-protector ![[LARGE]]
void private_array() { PrivateArray s; use(&s); }

//--- Base classes -------------------------------------------------------------

struct BasePointer { void *p; };
struct BaseBuffer { char b[16]; };
struct DerivedPointer : BasePointer { long long a, b; };
struct DerivedBuffer : BaseBuffer {};
struct DerivedPrivate : private PrivateData {};
struct HoldsDerivedPointer { DerivedPointer m; };

// Neither the pointer veto nor the access-control rule looks into a base class,
// so this is a GS buffer even though it inherits a pointer...
// CHECK-LABEL: @"?derived_pointer@@YAXXZ"
// CHECK:         alloca %struct.DerivedPointer, align 8, !stack-protector ![[LARGE]]
void derived_pointer() { DerivedPointer s; use(&s); }

// ...and so is an aggregate holding one as a member.
// CHECK-LABEL: @"?holds_derived_pointer@@YAXXZ"
// CHECK:         alloca %struct.HoldsDerivedPointer, align 8, !stack-protector ![[LARGE]]
void holds_derived_pointer() { HoldsDerivedPointer s; use(&s); }

// A private base with private members does not disqualify the derived class.
// CHECK-LABEL: @"?derived_private@@YAXXZ"
// CHECK:         alloca %struct.DerivedPrivate, align 8, !stack-protector ![[LARGE]]
void derived_private() { DerivedPrivate s; use(&s); }

// The search for a buffer does reach into a base class.
// CHECK-LABEL: @"?derived_buffer@@YAXXZ"
// CHECK:         alloca %struct.DerivedBuffer, align 1, !stack-protector ![[LARGE]]
void derived_buffer() { DerivedBuffer s; use(&s); }

//--- C++ specifics ------------------------------------------------------------

struct Polymorphic { virtual void f(); int a, b, c; };
struct Derived : PointerFree { int d; };
struct HoldsReference { int &r; int a, b; HoldsReference(int &x) : r(x) {} };

// The vptr is a pointer even though it is not a field, and it counts for a
// derived class too, where it physically lives in the base subobject.
// CHECK-LABEL: @"?polymorphic@@YAXXZ"
// CHECK:         alloca %struct.Polymorphic, align 8{{$}}
void polymorphic() { Polymorphic s; use(&s); }

// A base class is searched for a buffer just like a member.
// CHECK-LABEL: @"?derived@@YAXXZ"
// CHECK:         alloca %struct.Derived, align 4, !stack-protector ![[LARGE]]
void derived() { Derived s; use(&s); }

// A reference holds an address, so it disqualifies the aggregate just like a
// pointer does.
// CHECK-LABEL: @"?holds_reference@@YAXAEAH@Z"
// CHECK:         alloca %struct.HoldsReference, align 8{{$}}
void holds_reference(int &x) { HoldsReference s(x); use(&s); }

//--- _alloca ------------------------------------------------------------------

// A buffer allocated by _alloca is a GS buffer whatever its size. The element
// count that lets the backend recognise one does not survive a constant-sized
// alloca being folded to an array type, so it is marked here too.
// CHECK-LABEL: @"?alloca_large@@YAXXZ"
// CHECK:         alloca [64 x i8], align 16, !stack-protector ![[LARGE_NT:[0-9]+]]
void alloca_large() { use(_alloca(64)); }

// Below ssp-buffer-size, but still a GS buffer: _alloca has no threshold.
// CHECK-LABEL: @"?alloca_small@@YAXXZ"
// CHECK:         alloca [3 x i8], align 16, !stack-protector ![[SMALL_NT:[0-9]+]]
void alloca_small() { use(_alloca(3)); }

// CHECK-LABEL: @"?alloca_dynamic@@YAXH@Z"
// CHECK:         alloca i8, i64 %{{.*}}, align 16, !stack-protector ![[LARGE_NT]]
void alloca_dynamic(int n) { use(_alloca(n)); }

// CHECK-LABEL: @"?alloca_with_align@@YAXXZ"
// CHECK:         alloca [64 x i8], align 16, !stack-protector ![[LARGE_NT]]
void alloca_with_align() { use(__builtin_alloca_with_align(64, 128)); }

//--- Left to the backend ------------------------------------------------------

// A variable-length array lowers to an alloca with an element count, which the
// backend recognises as a GS buffer on its own.
// CHECK-LABEL: @"?vla@@YAXH@Z"
// CHECK:         alloca i8, i64 %{{[0-9]+}}, align 1{{$}}
void vla(int n) { char a[n]; use(a); }

//--- Opting out ---------------------------------------------------------------

// An explicit opt-out wins over the classification.
// CHECK-LABEL: @"?ignored@@YAXXZ"
// CHECK:         alloca [64 x i8], align 1, !stack-protector ![[IGNORE:[0-9]+]]
void ignored() {
  __attribute__((stack_protector_ignore)) char a[64];
  use(a);
}

// __declspec(safebuffers) turns the check off, so there is nothing to record.
// CHECK-LABEL: @"?safe@@YAXXZ"
// CHECK:         alloca [64 x i8], align 1{{$}}
__declspec(safebuffers) void safe() { char a[64]; use(a); }

// __declspec(strict_gs_check) asks for the broader "strong" heuristic, which
// the backend applies to the IR type itself.
// CHECK-LABEL: @"?strict@@YAXXZ"
// CHECK:         alloca [64 x i8], align 1{{$}}
__declspec(strict_gs_check) void strict() { char a[64]; use(a); }

// The second operand says the object is trivial, and so is one the backend may
// leave in the unguarded temporary area; see ms-stack-protector-gs-buffer-temp.
// CHECK-DAG: ![[LARGE]] = !{i32 2, i1 true}
// CHECK-DAG: ![[SMALL]] = !{i32 1, i1 true}
// CHECK-DAG: ![[LARGE_NT]] = !{i32 2}
// CHECK-DAG: ![[SMALL_NT]] = !{i32 1}
// CHECK-DAG: ![[IGNORE]] = !{i32 0}
