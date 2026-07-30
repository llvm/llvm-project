// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1y %s

// expected-no-diagnostics

// C++11 [basic.link]p6:
//   The name of a function declared in block scope and the name
//   of a variable declared by a block scope extern declaration
//   have linkage. If there is a visible declaration of an entity
//   with linkage having the same name and type, ignoring entities
//   declared outside the innermost enclosing namespace scope, the
//   block scope declaration declares that same entity and
//   receives the linkage of the previous declaration.

extern int same_entity;
constexpr int *get1() {
  int same_entity = 0; // not the same entity
  {
    extern int same_entity;
    return &same_entity;
  }
}
static_assert(get1() == &same_entity, "failed to find previous decl");

static int same_entity_2[3];
constexpr int *get2() {
  // This is a redeclaration of the same entity, even though it doesn't
  // inherit the type of the prior declaration.
  extern int same_entity_2[];
  return same_entity_2;
}
static_assert(get2() == same_entity_2, "failed to find previous decl");

static int different_entities;
constexpr int *get3() {
  int different_entities = 0;
  {
    // FIXME: This is not a redeclaration of the prior entity, because
    // it is not visible here. Under DR426, this is ill-formed, and without
    // it, the static_assert below should fail.
    extern int different_entities;
    return &different_entities;
  }
}
static_assert(get3() == &different_entities, "failed to find previous decl");
