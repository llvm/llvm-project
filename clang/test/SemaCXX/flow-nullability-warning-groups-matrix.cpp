// On/off matrix for the five -Wflow-nullable-* subgroups.
//
// flow-nullability-warning-groups.cpp only exercises the dereference subgroup.
// This file proves each of the five subgroups (dereference, arithmetic, return,
// assignment, argument) can be controlled INDEPENDENTLY: silencing one must not
// silence the others, the parent group silences all, and -Werror on one subgroup
// promotes only that one.
//
// All five fire by default:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value %s -verify=default
//
// Each -Wno-flow-nullable-SUB silences ONLY its own subgroup:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-flow-nullable-dereference %s -verify=noderef
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-flow-nullable-arithmetic %s -verify=noarith
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-flow-nullable-return %s -verify=noret
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-flow-nullable-assignment %s -verify=noassign
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-flow-nullable-argument %s -verify=noarg
//
// The parent group silences all five:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-flow-nullability %s -verify=noparent
//
// Each -Werror=flow-nullable-SUB promotes ONLY its own subgroup to an error,
// leaving the other four as warnings:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Werror=flow-nullable-dereference %s -verify=wederef
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Werror=flow-nullable-arithmetic %s -verify=wearith
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Werror=flow-nullable-return %s -verify=weret
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Werror=flow-nullable-assignment %s -verify=weassign
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Werror=flow-nullable-argument %s -verify=wearg

#pragma clang assume_nonnull begin

int *_Nullable getNullable();
void takesNonnull(int *_Nonnull p);

// --- dereference subgroup ---
void test_dereference(int *_Nullable p) {
  // noderef silences this one; every other config (including each -Werror=other)
  // still reports it as a warning. wederef promotes it to an error.
  *p = 1; // default-warning{{dereference of nullable pointer}} default-note{{add a null check}}
          // noarith-warning@-1{{dereference of nullable pointer}} noarith-note@-1{{add a null check}}
          // noret-warning@-2{{dereference of nullable pointer}} noret-note@-2{{add a null check}}
          // noassign-warning@-3{{dereference of nullable pointer}} noassign-note@-3{{add a null check}}
          // noarg-warning@-4{{dereference of nullable pointer}} noarg-note@-4{{add a null check}}
          // wederef-error@-5{{dereference of nullable pointer}} wederef-note@-5{{add a null check}}
          // wearith-warning@-6{{dereference of nullable pointer}} wearith-note@-6{{add a null check}}
          // weret-warning@-7{{dereference of nullable pointer}} weret-note@-7{{add a null check}}
          // weassign-warning@-8{{dereference of nullable pointer}} weassign-note@-8{{add a null check}}
          // wearg-warning@-9{{dereference of nullable pointer}} wearg-note@-9{{add a null check}}
}

// --- arithmetic subgroup ---
void test_arithmetic(int *_Nullable p) {
  (void)(p + 1); // default-warning{{pointer arithmetic on nullable pointer}} default-note{{add a null check}}
                 // noderef-warning@-1{{pointer arithmetic on nullable pointer}} noderef-note@-1{{add a null check}}
                 // noret-warning@-2{{pointer arithmetic on nullable pointer}} noret-note@-2{{add a null check}}
                 // noassign-warning@-3{{pointer arithmetic on nullable pointer}} noassign-note@-3{{add a null check}}
                 // noarg-warning@-4{{pointer arithmetic on nullable pointer}} noarg-note@-4{{add a null check}}
                 // wederef-warning@-5{{pointer arithmetic on nullable pointer}} wederef-note@-5{{add a null check}}
                 // wearith-error@-6{{pointer arithmetic on nullable pointer}} wearith-note@-6{{add a null check}}
                 // weret-warning@-7{{pointer arithmetic on nullable pointer}} weret-note@-7{{add a null check}}
                 // weassign-warning@-8{{pointer arithmetic on nullable pointer}} weassign-note@-8{{add a null check}}
                 // wearg-warning@-9{{pointer arithmetic on nullable pointer}} wearg-note@-9{{add a null check}}
}

// --- return subgroup ---
int *_Nonnull test_return(int *_Nullable p) {
  return p; // default-warning{{returning nullable pointer from function with nonnull return type}} default-note{{add a null check}}
            // noderef-warning@-1{{returning nullable pointer from function with nonnull return type}} noderef-note@-1{{add a null check}}
            // noarith-warning@-2{{returning nullable pointer from function with nonnull return type}} noarith-note@-2{{add a null check}}
            // noassign-warning@-3{{returning nullable pointer from function with nonnull return type}} noassign-note@-3{{add a null check}}
            // noarg-warning@-4{{returning nullable pointer from function with nonnull return type}} noarg-note@-4{{add a null check}}
            // wederef-warning@-5{{returning nullable pointer from function with nonnull return type}} wederef-note@-5{{add a null check}}
            // wearith-warning@-6{{returning nullable pointer from function with nonnull return type}} wearith-note@-6{{add a null check}}
            // weret-error@-7{{returning nullable pointer from function with nonnull return type}} weret-note@-7{{add a null check}}
            // weassign-warning@-8{{returning nullable pointer from function with nonnull return type}} weassign-note@-8{{add a null check}}
            // wearg-warning@-9{{returning nullable pointer from function with nonnull return type}} wearg-note@-9{{add a null check}}
}

// --- assignment subgroup ---
void test_assignment(int *_Nullable p) {
  int *_Nonnull q = p; // default-warning{{assigning nullable pointer to nonnull variable}} default-note{{add a null check}}
                       // noderef-warning@-1{{assigning nullable pointer to nonnull variable}} noderef-note@-1{{add a null check}}
                       // noarith-warning@-2{{assigning nullable pointer to nonnull variable}} noarith-note@-2{{add a null check}}
                       // noret-warning@-3{{assigning nullable pointer to nonnull variable}} noret-note@-3{{add a null check}}
                       // noarg-warning@-4{{assigning nullable pointer to nonnull variable}} noarg-note@-4{{add a null check}}
                       // wederef-warning@-5{{assigning nullable pointer to nonnull variable}} wederef-note@-5{{add a null check}}
                       // wearith-warning@-6{{assigning nullable pointer to nonnull variable}} wearith-note@-6{{add a null check}}
                       // weret-warning@-7{{assigning nullable pointer to nonnull variable}} weret-note@-7{{add a null check}}
                       // weassign-error@-8{{assigning nullable pointer to nonnull variable}} weassign-note@-8{{add a null check}}
                       // wearg-warning@-9{{assigning nullable pointer to nonnull variable}} wearg-note@-9{{add a null check}}
  (void)q;
}

// --- argument subgroup ---
void test_argument(int *_Nullable p) {
  takesNonnull(p); // default-warning{{passing nullable pointer to nonnull parameter}} default-note{{add a null check}}
                   // noderef-warning@-1{{passing nullable pointer to nonnull parameter}} noderef-note@-1{{add a null check}}
                   // noarith-warning@-2{{passing nullable pointer to nonnull parameter}} noarith-note@-2{{add a null check}}
                   // noret-warning@-3{{passing nullable pointer to nonnull parameter}} noret-note@-3{{add a null check}}
                   // noassign-warning@-4{{passing nullable pointer to nonnull parameter}} noassign-note@-4{{add a null check}}
                   // wederef-warning@-5{{passing nullable pointer to nonnull parameter}} wederef-note@-5{{add a null check}}
                   // wearith-warning@-6{{passing nullable pointer to nonnull parameter}} wearith-note@-6{{add a null check}}
                   // weret-warning@-7{{passing nullable pointer to nonnull parameter}} weret-note@-7{{add a null check}}
                   // weassign-warning@-8{{passing nullable pointer to nonnull parameter}} weassign-note@-8{{add a null check}}
                   // wearg-error@-9{{passing nullable pointer to nonnull parameter}} wearg-note@-9{{add a null check}}
}

// noparent silences everything: no diagnostics expected under that prefix.
// noparent-no-diagnostics

#pragma clang assume_nonnull end
