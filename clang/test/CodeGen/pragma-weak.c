// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - -verify | FileCheck %s

// CHECK: @weakvar = weak global
// CHECK: @__weakvar_alias = common global
// CHECK: @correct_linkage = weak global


// CHECK-DAG: @both = alias void (), void ()* @__both
// CHECK-DAG: @both2 = alias void (), void ()* @__both2
// CHECK-DAG: @weakvar_alias = weak alias i32, i32* @__weakvar_alias
// CHECK-DAG: @foo = weak alias void (), void ()* @__foo
// CHECK-DAG: @foo2 = weak alias void (), void ()* @__foo2
// CHECK-DAG: @stutter = weak alias void (), void ()* @__stutter
// CHECK-DAG: @stutter2 = weak alias void (), void ()* @__stutter2
// CHECK-DAG: @declfirst = weak alias void (), void ()* @__declfirst
// CHECK-DAG: @declfirstattr = weak alias void (), void ()* @__declfirstattr
// CHECK-DAG: @mix2 = weak alias void (), void ()* @__mix2
// CHECK-DAG: @a1 = weak alias void (), void ()* @__a1
// CHECK-DAG: @xxx = weak alias void (), void ()* @__xxx



// CHECK-LABEL: define weak void @weakdef()


#pragma weak weakvar
int weakvar;

#pragma weak weakdef
void weakdef(void) {}

#pragma weak param // expected-warning {{weak identifier 'param' never declared}}
#pragma weak correct_linkage
void f(int param) {
  int correct_linkage;
}

#pragma weak weakvar_alias = __weakvar_alias
int __weakvar_alias;

#pragma weak foo = __foo
void __foo(void) {}
// CHECK-LABEL: define void @__foo()


void __foo2(void) {}
#pragma weak foo2 = __foo2
// CHECK-LABEL: define void @__foo2()


///// test errors

#pragma weak unused // expected-warning {{weak identifier 'unused' never declared}}
#pragma weak unused_alias = __unused_alias  // expected-warning {{weak identifier '__unused_alias' never declared}}

#pragma weak td // expected-warning {{'weak' attribute only applies to variables and functions}}
typedef int td;

#pragma weak td2 = __td2 // expected-warning {{'weak' attribute only applies to variables and functions}}
typedef int __td2;

typedef int __td3;
#pragma weak td3 = __td3 // expected-warning {{'weak' attribute only applies to variables and functions}}

///// test weird cases

// test repeats

#pragma weak stutter = __stutter
#pragma weak stutter = __stutter
void __stutter(void) {}
// CHECK-LABEL: define void @__stutter()

void __stutter2(void) {}
#pragma weak stutter2 = __stutter2
#pragma weak stutter2 = __stutter2
// CHECK-LABEL: define void @__stutter2()


// test decl/pragma weak order

void __declfirst(void);
#pragma weak declfirst = __declfirst
void __declfirst(void) {}
// CHECK-LABEL: define void @__declfirst()

void __declfirstattr(void) __attribute((noinline));
#pragma weak declfirstattr = __declfirstattr
void __declfirstattr(void) {}
// CHECK-LABEL: define void @__declfirstattr()

//// test that other attributes are preserved

//// ensure that pragma weak/__attribute((weak)) play nice

void mix(void);
#pragma weak mix
__attribute((weak)) void mix(void) { }
// CHECK-LABEL: define weak void @mix()

// ensure following __attributes are preserved and that only a single
// alias is generated
#pragma weak mix2 = __mix2
void __mix2(void) __attribute((noinline));
void __mix2(void) __attribute((noinline));
void __mix2(void) {}
// CHECK-LABEL: define void @__mix2()

////////////// test #pragma weak/__attribute combinations

// if the SAME ALIAS is already declared then it overrides #pragma weak
// resulting in a non-weak alias in this case
void both(void) __attribute((alias("__both")));
#pragma weak both = __both
void __both(void) {}
// CHECK-LABEL: define void @__both()

// if the TARGET is previously declared then whichever aliasing method
// comes first applies and subsequent aliases are discarded.
// TODO: warn about this

void __both2(void);
void both2(void) __attribute((alias("__both2"))); // first, wins
#pragma weak both2 = __both2
void __both2(void) {}
// CHECK-LABEL: define void @__both2()

///////////// ensure that #pragma weak does not alter existing __attributes()

void __a1(void) __attribute((noinline));
#pragma weak a1 = __a1
void __a1(void) {}
// CHECK: define void @__a1() [[NI:#[0-9]+]]

#pragma weak xxx = __xxx
__attribute((pure,noinline,const)) void __xxx(void) { }
// CHECK: void @__xxx() [[RN:#[0-9]+]]

///////////// PR10878: Make sure we can call a weak alias
void SHA512Pad(void *context) {}
#pragma weak SHA384Pad = SHA512Pad
void PR10878() { SHA384Pad(0); }
// CHECK: call void @SHA384Pad(i8* null)


// PR14046: Parse #pragma weak in function-local context
extern int PR14046e(void);
void PR14046f() {
#pragma weak PR14046e
  PR14046e();
}
// CHECK: declare extern_weak i32 @PR14046e()

// Parse #pragma weak after a label or case statement
extern int PR16705a(void);
extern int PR16705b(void);
extern int PR16705c(void);
void PR16705f(int a) {
  switch(a) {
  case 1:
#pragma weak PR16705a
    PR16705a();
  default:
#pragma weak PR16705b
    PR16705b();
  }
label:
  #pragma weak PR16705c
  PR16705c();
}

// CHECK: declare extern_weak i32 @PR16705a()
// CHECK: declare extern_weak i32 @PR16705b()
// CHECK: declare extern_weak i32 @PR16705c()


///////////// TODO: stuff that still doesn't work

// due to the fact that disparate TopLevelDecls cannot affect each other
// (due to clang's Parser and ASTConsumer behavior, and quite reasonable)
// #pragma weak must appear before or within the same TopLevelDecl as it
// references.
void yyy(void){}
void zzz(void){}
#pragma weak yyy
// NOTE: weak doesn't apply, not before or in same TopLevelDec(!)
// CHECK-LABEL: define void @yyy()

int correct_linkage;

// CHECK: attributes [[NI]] = { noinline nounwind{{.*}} }
// CHECK: attributes [[RN]] = { noinline nounwind optnone readnone{{.*}} }
