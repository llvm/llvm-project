// RUN: %clang_analyze_cc1 -triple thumbv7-apple-ios0.0.0 -target-feature +neon -analyzer-checker=core -verify -fblocks -Wno-objc-root-class %s
// expected-no-diagnostics

// Handle casts of vectors to structs, and loading a value.
typedef float float32_t;
typedef __attribute__((neon_vector_type(2))) float32_t float32x2_t;

typedef struct
{
    float x, y;
} Rdar11405978Vec;
    
float32x2_t rdar11405978_bar(void);
float32_t rdar11405978(void) {
  float32x2_t v = rdar11405978_bar();
  Rdar11405978Vec w = *(Rdar11405978Vec *)&v;
  return w.x; // no-warning
}
