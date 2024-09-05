// RUN: %clang_cc1 -verify -fsyntax-only --embed-dir=%S/Inputs -std=c2x %s -Wno-constant-logical-operand
// RUN: %clang_cc1 -verify -fsyntax-only --embed-dir=%S/Inputs -std=c2x %s -Wno-constant-logical-operand -fexperimental-new-constant-interpreter

/* WG14 N3017: full
 * #embed - a scannable, tooling-friendly binary resource inclusion mechanism
 */

// C23 6.10p6
char b1[] = {
#embed "boop.h" limit(5)
,
#embed "boop.h" __limit__(5)
};

// C23 6.10.1p19
#if __has_embed(__FILE__ ext::token(0xB055))
#error "Supports an extension parameter Clang never claimed to support?"
#endif

#if !__has_embed(__FILE__ clang::offset(0))
#error "Doesn't support an extension Clang claims to support?"
#endif

// C23 6.10.1p20
void parse_into_s(short* ptr, unsigned char* ptr_bytes, unsigned long long size);
int f() {
#if __has_embed ("bits.bin" ds9000::element_type(short))
  /* Implementation extension: create short integers from the */
  /* translation environment resource into */
  /* a sequence of integer constants */
  short meow[] = {
#embed "bits.bin" ds9000::element_type(short)
  };
#elif __has_embed ("bits.bin")
  /* no support for implementation-specific */
  /* ds9000::element_type(short) parameter */
  unsigned char meow_bytes[] = {
  #embed "bits.bin"
  };
  short meow[sizeof(meow_bytes) / sizeof(short)] = {};
  /* parse meow_bytes into short values by-hand! */
  parse_into_s(meow, meow_bytes, sizeof(meow_bytes));
#else
#error "cannot find bits.bin resource"
#endif
  return (int)(meow[0] + meow[(sizeof(meow) / sizeof(*meow)) - 1]);
}

// NOTE: we don't have a good way to test infinite resources from within lit.
int g() {
#if __has_embed(<infinite-resource> limit(0)) == 2
  // if <infinite-resource> exists, this
  // token sequence is always taken.
  return 0;
#else
  // the ’infinite-resource’ resource does not exist
  #error "The resource does not exist"
#endif
  // expected-error@-2 {{"The resource does not exist"}}
}

#include <stddef.h>
void have_you_any_wool(const unsigned char*, size_t);
int h() {
  static const unsigned char baa_baa[] = {
#embed __FILE__
  };
  have_you_any_wool(baa_baa, sizeof(baa_baa));
  return 0;
}

// C23 6.10.3.1p17: not tested here because we do not currently support any
// platforms where CHAR_BIT != 8.

// C23 6.10.3.1p18
int i() {
/* Braces may be kept or elided as per normal initialization rules */
  int i = {
#embed "i.dat"
  }; /* valid if i.dat produces 1 value,
        i value is [0, 2(embed element width)) */
  int i2 =
#embed "i.dat"
  ; /* valid if i.dat produces 1 value,
       i2 value is [0, 2(embed element width)) */
  struct s {
    double a, b, c;
    struct { double e, f, g; };
    double h, i, j;
  };
  struct s x = {
    /* initializes each element in order according to initialization
    rules with comma-separated list of integer constant expressions
    inside of braces */
    #embed "s.dat"
  };
  return 0;
}

// C23 6.10.3.1p19: not tested here because it's a runtime test rather than one
// which can be handled at compile time (it validates file contents via fread).

// C23 6.10.3.2p5
int j() {
  static const char sound_signature[] = {
#embed <jump.wav> limit(2+2)
  };
  static_assert((sizeof(sound_signature) / sizeof(*sound_signature)) == 4,
    "There should only be 4 elements in this array.");
  // verify PCM WAV resource
  static_assert(sound_signature[0] == 'R');
  static_assert(sound_signature[1] == 'I');
  static_assert(sound_signature[2] == 'F');
  static_assert(sound_signature[3] == 'F');
  static_assert(sizeof(sound_signature) == 4);
  return 0;
}

// C23 6.10.3p6
int k() {
#define TWO_PLUS_TWO 2+2
  static const char sound_signature[] = {
#embed <jump.wav> limit(TWO_PLUS_TWO)
  };
  static_assert((sizeof(sound_signature) / sizeof(*sound_signature)) == 4,
    "There should only be 4 elements in this array.");
  // verify PCM WAV resource
  static_assert(sound_signature[0] == 'R');
  static_assert(sound_signature[1] == 'I');
  static_assert(sound_signature[2] == 'F');
  static_assert(sound_signature[3] == 'F');
  static_assert(sizeof(sound_signature) == 4);
  return 0;
}

// C23 6.10.3.2p7: not tested here because we do not currently support any
// platforms where CHAR_BIT != 8.

// C23 6.10.3.2p8: not tested here because it requires access to an infinite
// resource like /dev/urandom.

// C23 6.10.3.3p4
char *strcpy(char *, const char *);
#ifndef SHADER_TARGET
  #define SHADER_TARGET "bits.bin"
#endif
extern char* null_term_shader_data;
void fill_in_data () {
  const char internal_data[] = {
#embed SHADER_TARGET \
  suffix(,)
  0
  };
  strcpy(null_term_shader_data, internal_data);
}

// C23 6.10.3.4p4
#ifndef SHADER_TARGET
#define SHADER_TARGET "bits.bin"
#endif
extern char* merp;
void init_data () {
  const char whl[] = {
#embed SHADER_TARGET \
    prefix(0xEF, 0xBB, 0xBF, ) /* UTF-8 BOM */ \
    suffix(,)
    0
  };
  // always null terminated,
  // contains BOM if not-empty
  const int is_good = (sizeof(whl) == 1 && whl[0] == '\0')
    || (whl[0] == '\xEF' && whl[1] == '\xBB'
    && whl[2] == '\xBF' && whl[sizeof(whl) - 1] == '\0');
  static_assert(is_good);
  strcpy(merp, whl);
}

// C23 6.10.3.5p3
int l() {
  return
#embed <bits.bin> limit(0) prefix(1) if_empty(0)
  ;
  // becomes:
  // return 0;

  // Validating the assumption from the example in the standard.
  static_assert(
#embed <bits.bin> limit(0) prefix(1) if_empty(0)
    == 0);
}

// C23 6.10.3.5p4
void fill_in_data_again() {
  const char internal_data[] = {
#embed SHADER_TARGET \
  suffix(, 0) \
  if_empty(0)
  };
  strcpy(null_term_shader_data, internal_data);
}

// C23 6.10.3.5p5
int m() {
  return
#embed __FILE__ limit(0) if_empty(45540)
  ;

  // Validating the assumption from the example in the standard.
  static_assert(
#embed __FILE__ limit(0) if_empty(45540)
    == 45540);
}

// 6.10.9.1p1
static_assert(__STDC_EMBED_NOT_FOUND__ == 0);
static_assert(__STDC_EMBED_FOUND__ == 1);
static_assert(__STDC_EMBED_EMPTY__ == 2);
