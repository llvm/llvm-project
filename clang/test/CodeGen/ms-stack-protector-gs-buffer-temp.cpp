// Check which of the stack objects clang materialises itself clang-cl's default
// /GS classifies. cl.exe analyses the objects the source declares, but gives an
// anonymous ABI slot holding a trivial object a frame slot outside the guarded
// region, and so leaves it out.
//
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -O1 \
// RUN:     -disable-llvm-passes -stack-protector 4 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

struct Buf { char b[16]; };
struct Ctor { char b[16]; Ctor(); };
struct Dtor { char b[16]; ~Dtor(); };
struct CopyCtor { char b[16]; CopyCtor(); CopyCtor(const CopyCtor &); };
struct NoCopy { char b[16]; NoCopy(); NoCopy(NoCopy &&); };

Buf makeBuf();
Ctor makeCtor();
Dtor makeDtor();
void takeBuf(Buf);
void takeCtor(Ctor);
void takeCopyCtor(CopyCtor);
void takeNoCopy(NoCopy);
void use(const void *);

//--- Indirect return slots ----------------------------------------------------

// The slot for a discarded trivial return value is anonymous storage, so it is
// not classified at all.
// CHECK-LABEL: @"?discard_trivial@@YAXXZ"
// CHECK:         alloca %struct.Buf, align 1{{$}}
void discard_trivial() { makeBuf(); }

// Once the type needs a constructor or a destructor to run, the slot holds a
// real object and cl.exe covers it.
// CHECK-LABEL: @"?discard_dtor@@YAXXZ"
// CHECK:         alloca %struct.Dtor, align 1, !stack-protector ![[LARGE:[0-9]+]]
void discard_dtor() { makeDtor(); }

// A temporary the source binds to a reference is a declared object, so it is
// classified even when trivial. The second operand lets the backend drop it
// again if its address turns out to be used for nothing but the return value;
// see llvm/test/CodeGen/X86/stack-protector-gs-buffer.ll.
// CHECK-LABEL: @"?bind_trivial@@YAXXZ"
// CHECK:         alloca %struct.Buf, align 1, !stack-protector ![[LARGE_TRIV:[0-9]+]]
void bind_trivial() { const Buf &r = makeBuf(); use(&r); }

// Likewise a declared local, which clang gives to the callee directly.
// CHECK-LABEL: @"?local_from_call@@YAXXZ"
// CHECK:         alloca %struct.Buf, align 1, !stack-protector ![[LARGE_TRIV]]
void local_from_call() { Buf b = makeBuf(); use(&b); }

//--- Outgoing by-value arguments ----------------------------------------------

// cl.exe builds the copy in the outgoing argument area, which the cookie does
// not reach, so a trivial argument is not classified.
// CHECK-LABEL: @"?pass_trivial@@YAXXZ"
// CHECK:         alloca %struct.Buf, align 1{{$}}
void pass_trivial() { Buf b; takeBuf(b); }

// With a non-trivial type it builds a real object first. It can only then copy
// that into the argument area, and so only then benefits from the guard, if the
// copy is bitwise.
// CHECK-LABEL: @"?pass_ctor@@YAXXZ"
// CHECK:         alloca %struct.Ctor, align 1, !stack-protector ![[LARGE]]
void pass_ctor() { takeCtor(Ctor()); }

// A user-provided copy constructor has to run on the argument area itself.
// CHECK-LABEL: @"?pass_copy_ctor@@YAXXZ"
// CHECK:         alloca %struct.CopyCtor, align 1{{$}}
void pass_copy_ctor() { takeCopyCtor(CopyCtor()); }

// A deleted copy constructor is no better.
// CHECK-LABEL: @"?pass_no_copy@@YAXXZ"
// CHECK:         alloca %struct.NoCopy, align 1{{$}}
void pass_no_copy() { takeNoCopy(NoCopy()); }

//--- By-value parameters ------------------------------------------------------

// A by-value parameter lives in the caller's frame, where the guard does not
// reach, so clang relocates a vulnerable one into a copy of its own. This
// mirrors the `<name>$GSCopy$` slot cl.exe emits.
// CHECK-LABEL: @"?param@@YAXUBuf@@@Z"
// CHECK:         %[[COPY:["a-zA-Z0-9$]+]] = alloca %struct.Buf, align 1, !stack-protector ![[LARGE]]
// CHECK:         call void @llvm.memcpy{{.*}}(ptr {{[^,]*}}%[[COPY]],
// CHECK:         call void @"?use@@YAXPEBX@Z"(ptr {{[^,]*}}%[[COPY]])
void param(Buf b) { use(&b); }

// A parameter that is not a GS buffer is left where it is.
// CHECK-LABEL: @"?param_int@@YAXH@Z"
// CHECK-NOT:     !stack-protector
// CHECK:       }
void param_int(int n) { use(&n); }

// CHECK-DAG: ![[LARGE]] = !{i32 2}
// CHECK-DAG: ![[LARGE_TRIV]] = !{i32 2, i1 true}
