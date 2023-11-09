// RUN: %clang_cc1 %s -std=c23 -E -embed-dir=%S/Inputs | FileCheck %s

// Ensure that we print out the correct data to the preprocessed file. Note,
// #embed will do a base64 encoding of the file contents, so if art.txt changes,
// this test will need to change accordingly as well.
const char data[] = {
#embed <media/art.txt>
};

// CHECK: # 1 "<built-in:embed:1>" 1
// CHECK-NEXT: __builtin_pp_embed(unsigned char,"{{.*}}media{{\\|/}}art.txt","ICAgICAgICAgICBfXyAgXwogICAgICAgLi0uJyAgYDsgYC0uXyAgX18gIF8KICAgICAgKF8sICAgICAgICAgLi06JyAgYDsgYC0uXwogICAgLCdvIiggICAgICAgIChfLCAgICAgICAgICAgKQogICAoX18sLScgICAgICAsJ28iKCAgICAgICAgICAgICk+CiAgICAgICggICAgICAgKF9fLC0nICAgICAgICAgICAgKQogICAgICAgYC0nLl8uLS0uXyggICAgICAgICAgICAgKQogICAgICAgICAgfHx8ICB8fHxgLScuXy4tLS5fLi0nCiAgICAgICAgICAgICAgICAgICAgIHx8fCAgfHx8Cg==")
// CHECK-NEXT: # 8 "{{.*}}embed_preprocess_to_file.c" 2
