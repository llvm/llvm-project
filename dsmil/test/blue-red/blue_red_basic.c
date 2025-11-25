/**
 * @file blue_red_basic.c
 * @brief Basic blue vs red build tests
 *
 * RUN: dsmil-clang -fdsmil-role=blue -S -emit-llvm %s -o - | \
 * RUN:   FileCheck %s --check-prefix=BLUE
 *
 * RUN: dsmil-clang -fdsmil-role=red -S -emit-llvm %s -o - | \
 * RUN:   FileCheck %s --check-prefix=RED
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <dsmil_attributes.h>

// Test 1: Red team hook
// RED: call {{.*}} @dsmil_red_log
// BLUE-NOT: call {{.*}} @dsmil_red_log
DSMIL_RED_TEAM_HOOK("test_hook")
void test_red_hook(void) {
    int x = 42;
}

// Test 2: Attack surface
// RED: !dsmil.attack_surface
// BLUE: define {{.*}} @test_attack_surface
DSMIL_ATTACK_SURFACE
void test_attack_surface(const char *input) {
    (void)input;
}

// Test 3: Vulnerability injection
// RED: call {{.*}} @dsmil_red_scenario
DSMIL_VULN_INJECT("buffer_overflow")
void test_vuln_inject(char *dest, const char *src) {
    (void)dest;
    (void)src;
}

// Test 4: Blast radius
DSMIL_BLAST_RADIUS
void test_blast_radius(void) {
    int x = 0;
}

// Test 5: Build role
DSMIL_BUILD_ROLE("blue")
int main(void) {
    test_red_hook();
    test_attack_surface("test");
    test_vuln_inject(0, 0);
    test_blast_radius();
    return 0;
}
