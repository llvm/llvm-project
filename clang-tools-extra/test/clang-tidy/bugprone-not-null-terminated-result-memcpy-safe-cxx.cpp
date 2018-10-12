// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -std=c++11

#define __STDC_LIB_EXT1__ 1
#define __STDC_WANT_LIB_EXT1__ 1

namespace std {
template <typename T>
struct basic_string {
  basic_string();
  const T *data() const;
  unsigned long size() const;
};
typedef basic_string<char> string;
}
typedef unsigned int size_t;
typedef int errno_t;
size_t strlen(const char *);
void *malloc(size_t);
void *realloc(void *, size_t);

template <size_t size>
errno_t strncpy_s(char (&dest)[size], const char *src, size_t length);
errno_t strncpy_s(char *, size_t, const char *, size_t);

template <size_t size>
char *strncpy(char (&dest)[size], const char *src, size_t length);
char *strncpy(char *, const char *, size_t);

template <size_t size>
errno_t strcpy_s(char (&dest)[size], const char *);
errno_t strcpy_s(char *, size_t, const char *);

template <size_t size>
char *strcpy(char (&dest)[size], const char *);
char *strcpy(char *, const char *);

errno_t memcpy_s(void *, size_t, const void *, size_t);
void *memcpy(void *, const void *, size_t);


//===----------------------------------------------------------------------===//
// memcpy() - destination array tests
//===----------------------------------------------------------------------===//

void bad_memcpy_not_just_char_dest(const char *src) {
  unsigned char dest00[13];
  memcpy(dest00, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: unsigned char dest00[14];
  // CHECK-FIXES-NEXT: memcpy_s(dest00, 14, src, strlen(src) + 1);
}

void good_memcpy_not_just_char_dest(const char *src) {
  unsigned char dst00[14];
  memcpy_s(dst00, 14, src, strlen(src) + 1);
}

void bad_memcpy_known_dest(const char *src) {
  char dest01[13];
  memcpy(dest01, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: dest01[14];
  // CHECK-FIXES-NEXT: strcpy_s(dest01, src);
}

void good_memcpy_known_dest(const char *src) {
  char dst01[14];
  strcpy_s(dst01, src);
}

//===----------------------------------------------------------------------===//
// memcpy() - length tests
//===----------------------------------------------------------------------===//

void bad_memcpy_full_source_length(std::string src) {
  char *dest20;
  dest20 = reinterpret_cast<char *>(malloc(src.size()));
  memcpy(dest20, src.data(), src.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: dest20 = reinterpret_cast<char *>(malloc(src.size() + 1));
  // CHECK-FIXES-NEXT: strcpy(dest20, src.data());
}

void good_memcpy_full_source_length(std::string src) {
  char dst20[14];
  strcpy_s(dst20, src.data());
}

void bad_memcpy_partial_source_length(const char *src) {
  char dest21[13];
  memcpy(dest21, src, strlen(src) - 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest21[14];
  // CHECK-FIXES-NEXT: strncpy_s(dest21, src, strlen(src) - 1);
}

void good_memcpy_partial_source_length(const char *src) {
  char dst21[14];
  strncpy_s(dst21, src, strlen(src) - 1);
}


//===----------------------------------------------------------------------===//
// memcpy_s() - destination array tests
//===----------------------------------------------------------------------===//

void bad_memcpy_s_unknown_dest(char *dest40, const char *src) {
  memcpy_s(dest40, 13, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strcpy_s(dest40, 13, src);
}

void good_memcpy_s_unknown_dest(char *dst40, const char *src) {
  strcpy_s(dst40, 13, src);
}

void bad_memcpy_s_known_dest(const char *src) {
  char dest41[13];
  memcpy_s(dest41, 13, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest41[14];
  // CHECK-FIXES: strcpy_s(dest41, src);
}

void good_memcpy_s_known_dest(const char *src) {
  char dst41[14];
  strcpy_s(dst41, src);
}

//===----------------------------------------------------------------------===//
// memcpy_s() - length tests
//===----------------------------------------------------------------------===//

void bad_memcpy_s_full_source_length(const char *src) {
  char dest60[13];
  memcpy_s(dest60, 13, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest60[14];
  // CHECK-FIXES-NEXT: strcpy_s(dest60, src);
}

void good_memcpy_s_full_source_length(const char *src) {
  char dst60[14];
  strcpy_s(dst60, src);
}

void bad_memcpy_s_partial_source_length(const char *src) {
  char dest61[13];
  memcpy_s(dest61, 13, src, strlen(src) - 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest61[14];
  // CHECK-FIXES-NEXT: strncpy_s(dest61, src, strlen(src) - 1);
}

void good_memcpy_s_partial_source_length(const char *src) {
  char dst61[14];
  strncpy_s(dst61, src, strlen(src) - 1);
}

