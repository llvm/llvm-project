// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct nested
{
  union {
    const char *single;
    const char *const *multi;
  } output;
};
static const char * const test[] = {
  "test",
};
const struct nested data[] =
{
    {
        {
            .multi = test,
        },
    },
    {
        {
            .single = "hello",
        },
    },
};

// LLVM: @data = constant { { { ptr } }, { { ptr } } }
