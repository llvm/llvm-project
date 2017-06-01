// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0

global clk_event_t ce; // expected-error {{the '__global clk_event_t' type cannot be used to declare a program scope variable}}
