// RUN: %clang_cc1 -std=c++1z -fmodules-ts %s -verify
// expected-no-diagnostics
export module M;

export int external_linkage_var;
int module_linkage_var;
static int internal_linkage_var;

export void external_linkage_fn() {}
void module_linkage_fn() {}
static void internal_linkage_fn() {}

export struct external_linkage_class {};
struct module_linkage_class {};
namespace {
struct internal_linkage_class {};
} // namespace

void use() {
  external_linkage_fn();
  module_linkage_fn();
  internal_linkage_fn();
  (void)external_linkage_class{};
  (void)module_linkage_class{};
  (void)internal_linkage_class{};
  (void)external_linkage_var;
  (void)module_linkage_var;
  (void)internal_linkage_var;
}
