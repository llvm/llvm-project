// RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only -fno-spell-checking -verify %s

typedef int v4si __attribute__((vector_size(16)));
typedef float v4f __attribute__((vector_size(16)));
typedef double v2d __attribute__((vector_size(16)));
typedef long long v2ll __attribute__((vector_size(16)));
typedef long long v4ll __attribute__((vector_size(32)));
typedef long long v8ll __attribute__((vector_size(64)));
void call_x86_64_builtins(void) {
  unsigned long long *ullp;
  void *vp;
  v4f vec4floats;
  v2d vec2doubles;
  v2ll vec2longlongs;
  v4ll vec4longlongs;
  v8ll vec8longlongs;
  (void)__builtin_ia32_readeflags_u64();                             // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_writeeflags_u64(4);                           // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_cvtss2si64(vec4floats);                       // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_cvttss2si64(vec4floats);                      // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_cvtsd2si64(vec2doubles);                      // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_cvttsd2si64(vec2doubles);                     // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_crc32di(4, 4);                                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_rdfsbase64();                                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_rdgsbase64();                                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_wrfsbase64(4);                                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_wrgsbase64(4);                                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_fxrstor64(vp);                                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_fxsave64(vp);                                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_xsave64(vp, 4);                               // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_xrstor64(vp, 4);                              // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_xsaveopt64(vp, 4);                            // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_xrstors64(vp, 4);                             // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_xsavec64(vp, 4);                              // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_xsaves64(vp, 4);                              // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_addcarryx_u64(4, 4, 4, ullp);                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_addcarry_u64(4, 4, 4, ullp);                  // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_subborrow_u64(4, 4, 4, ullp);                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_rdseed64_step(ullp);                          // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_bextr_u64(4, 4);                              // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_bzhi_di(4, 4);                                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_pdep_di(4, 4);                                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_pext_di(4, 4);                                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_bextri_u64(4, 4);                             // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_pbroadcastq512_gpr_mask(4, vec8longlongs, 4); // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_pbroadcastq128_gpr_mask(4, vec2longlongs, 4); // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_pbroadcastq256_gpr_mask(4, vec4longlongs, 4); // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_vcvtsd2si64(vec2doubles, 4);                  // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_vcvtsd2usi64(vec2doubles, 4);                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_vcvtss2si64(vec4floats, 4);                   // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_vcvtss2usi64(vec4floats, 4);                  // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_vcvttsd2si64(vec2doubles, 4);                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_vcvttsd2usi64(vec2doubles, 4);                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_vcvttss2si64(vec4floats, 4);                  // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_vcvttss2usi64(vec4floats, 4);                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_cvtsi2sd64(vec2doubles, 4, 4);                // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_cvtsi2ss64(vec4floats, 4, 4);                 // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_cvtusi2sd64(vec2doubles, 4, 4);               // expected-error{{use of unknown builtin}}
  (void)__builtin_ia32_cvtusi2ss64(vec4floats, 4, 4);                // expected-error{{use of unknown builtin}}
}
