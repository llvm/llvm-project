// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Wno-deprecated-builtins -I %S/Inputs %s -verify -std=c++11
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Wno-deprecated-builtins -I %S/Inputs %s -ast-dump -ast-dump-filter merge -std=c++11 | FileCheck %s

// expected-no-diagnostics

void use_implicit_new() { operator new[](3); }

@import dummy;
@import cxx_decls.imported;

void test_delete(int *p) {
  // We can call the normal global deallocation function even though it has only
  // ever been explicitly declared in an unimported submodule.
  delete p;
}

void friend_1(HasFriends s) {
  s.private_thing();
}
void test_friends(HasFriends s) {
  friend_1(s);
  friend_2(s);
}

static_assert(!__is_trivial(HasNontrivialDefaultConstructor), "");
static_assert(!__has_trivial_constructor(HasNontrivialDefaultConstructor), "");

void use_implicit_new_again() { operator new[](3); }

int importMergeUsedFlag = getMergeUsedFlag();

int use_name_for_linkage(NameForLinkage &nfl) {
  return nfl.n + nfl.m;
}
int use_overrides_virtual_functions(OverridesVirtualFunctions ovf) { return 0; }

@import cxx_decls_merged;

NameForLinkage2Inner use_name_for_linkage2_inner;
NameForLinkage2 use_name_for_linkage2;

int name_for_linkage_test = use_name_for_linkage(name_for_linkage);
int overrides_virtual_functions_test =
    use_overrides_virtual_functions(overrides_virtual_functions);

void use_extern_c_function() { ExternCFunction(); }

int use_namespace_alias() { return Alias::a + Alias::b; }

@import cxx_decls_premerged;

void use_extern_c_function_2() { ExternCFunction(); }

InhCtorB inhctorb(2);

// CHECK: VarDecl [[mergeUsedFlag:0x[0-9a-f]*]] {{.*}} in cxx_decls.imported used mergeUsedFlag
// CHECK: VarDecl {{0x[0-9a-f]*}} prev [[mergeUsedFlag]] {{.*}} in cxx_decls_merged used mergeUsedFlag
