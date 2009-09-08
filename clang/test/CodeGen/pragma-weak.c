// RUN: clang-cc -emit-llvm %s -o - -verify | FileCheck %s

// CHECK: @weakvar = weak global
// CHECK: @__weakvar_alias = common global
// CHECK: @correct_linkage = weak global


// CHECK: @both = alias void ()* @__both
// CHECK: @both2 = alias void ()* @__both2
// CHECK: @both3 = alias weak void ()* @__both3
// CHECK: @a3 = alias weak void ()* @__a3
// CHECK: @weakvar_alias = alias weak i32* @__weakvar_alias
// CHECK: @foo = alias weak void ()* @__foo
// CHECK: @foo2 = alias weak void ()* @__foo2
// CHECK: @stutter = alias weak void ()* @__stutter
// CHECK: @stutter2 = alias weak void ()* @__stutter2
// CHECK: @declfirst = alias weak void ()* @__declfirst
// CHECK: @declfirstattr = alias weak void ()* @__declfirstattr
// CHECK: @mix2 = alias weak void ()* @__mix2
// CHECK: @a1 = alias weak void ()* @__a1
// CHECK: @xxx = alias weak void ()* @__xxx



// CHECK: define weak void @weakdef()


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
// CHECK: define void @__foo()


void __foo2(void) {}
#pragma weak foo2 = __foo2
// CHECK: define void @__foo2()


///// test errors

#pragma weak unused // expected-warning {{weak identifier 'unused' never declared}}
#pragma weak unused_alias = __unused_alias  // expected-warning {{weak identifier '__unused_alias' never declared}}

#pragma weak td // expected-warning {{weak identifier 'td' never declared}}
typedef int td;

#pragma weak td2 = __td2 // expected-warning {{weak identifier '__td2' never declared}}
typedef int __td2;


///// test weird cases

// test repeats

#pragma weak stutter = __stutter
#pragma weak stutter = __stutter
void __stutter(void) {}
// CHECK: define void @__stutter()

void __stutter2(void) {}
#pragma weak stutter2 = __stutter2
#pragma weak stutter2 = __stutter2
// CHECK: define void @__stutter2()


// test decl/pragma weak order

void __declfirst(void);
#pragma weak declfirst = __declfirst
void __declfirst(void) {}
// CHECK: define void @__declfirst()

void __declfirstattr(void) __attribute((noinline));
#pragma weak declfirstattr = __declfirstattr
void __declfirstattr(void) {}
// CHECK: define void @__declfirstattr()

//// test that other attributes are preserved

//// ensure that pragma weak/__attribute((weak)) play nice

void mix(void);
#pragma weak mix
__attribute((weak)) void mix(void) { }
// CHECK: define weak void @mix()

// ensure following __attributes are preserved and that only a single
// alias is generated
#pragma weak mix2 = __mix2
void __mix2(void) __attribute((noinline));
void __mix2(void) __attribute((noinline));
void __mix2(void) {}
// CHECK: define void @__mix2()

////////////// test #pragma weak/__attribute combinations

// if the SAME ALIAS is already declared then it overrides #pragma weak
// resulting in a non-weak alias in this case
void both(void) __attribute((alias("__both")));
#pragma weak both = __both
void __both(void) {}
// CHECK: define void @__both()

// if the TARGET is previously declared then whichever aliasing method
// comes first applies and subsequent aliases are discarded.
// TODO: warn about this

void __both2(void);
void both2(void) __attribute((alias("__both2"))); // first, wins
#pragma weak both2 = __both2
void __both2(void) {}
// CHECK: define void @__both2()

void __both3(void);
#pragma weak both3 = __both3 // first, wins
void both3(void) __attribute((alias("__both3")));
void __both3(void) {}
// CHECK: define void @__both3()

///////////// ensure that #pragma weak does not alter existing __attributes()

void __a1(void) __attribute((noinline));
#pragma weak a1 = __a1
void __a1(void) {}
// CHECK: define void @__a1()

// attributes introduced BEFORE a combination of #pragma weak and alias()
// hold...
void __a3(void) __attribute((noinline));
#pragma weak a3 = __a3
void a3(void) __attribute((alias("__a3")));
void __a3(void) {}
// CHECK: define void @__a3()

#pragma weak xxx = __xxx
__attribute((pure,noinline,const,fastcall)) void __xxx(void) { }
// CHECK: void @__xxx()

/// TODO: stuff that still doesn't work

// due to the fact that disparate TopLevelDecls cannot affect each other
// (due to clang's Parser and ASTConsumer behavior, and quite reasonable)
// #pragma weak must appear before or within the same TopLevelDecl as it
// references.
void yyy(void){}
void zzz(void){}
#pragma weak yyy
// NOTE: weak doesn't apply, not before or in same TopLevelDec(!)
// CHECK: define void @yyy()

int correct_linkage;
