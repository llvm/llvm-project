// RUN: %clang_cc1 %s -emit-llvm -o /dev/null
// PR4332

static int highly_aligned __attribute__((aligned(4096)));

int f() {
	return highly_aligned;
}
