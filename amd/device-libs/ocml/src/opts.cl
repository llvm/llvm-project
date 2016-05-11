
// Default opt controls for the library

__attribute__((always_inline, const, weak)) int __ocml_have_fast_fma32(void) { return 0; }
__attribute__((always_inline, const, weak)) int __ocml_have_fast_fma64(void) { return 1; }
__attribute__((always_inline, const, weak)) int __ocml_finite_only_opt(void) { return 0; }
__attribute__((always_inline, const, weak)) int __ocml_fast_relaxed_opt(void) { return 0; }
__attribute__((always_inline, const, weak)) int __ocml_daz_opt(void) { return 1; }
__attribute__((always_inline, const, weak)) int __ocml_amd_opt(void) { return 1; }
__attribute__((always_inline, const, weak)) int __ocml_correctly_rounded_sqrt32(void) { return 0; }

