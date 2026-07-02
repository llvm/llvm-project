// RUN: %check_clang_tidy %s bugprone-suspicious-fread-fwrite-return %t

typedef decltype(sizeof(int)) size_t;

struct FILE;
extern "C" size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
extern "C" size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);

void test_direct_comparison(FILE *fp, void *buf, size_t size) {
  if (fwrite(buf, 1, size, fp) < -1) {
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: return value of 'fwrite' is an unsigned 'size_t'; this comparison is always false [bugprone-suspicious-fread-fwrite-return]
    // CHECK-FIXES: if (fwrite(buf, 1, size, fp) != size) {
  }

  if (fwrite(buf, 1, size, fp) >= -1) {
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: return value of 'fwrite' is an unsigned 'size_t'; this comparison is always true
    // CHECK-FIXES: if (fwrite(buf, 1, size, fp) != size) {
  }

  if (fwrite(buf, 1, size, fp) <= 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: suspicious comparison against 0; 'fwrite' returns an unsigned 'size_t', so comparing it with '<= 0' is equivalent to comparing it with '== 0'. To detect short reads or writes, compare against the 'nmemb' argument
    // CHECK-FIXES: if (fwrite(buf, 1, size, fp) != size) {
  }

  if (0 >= fread(buf, 1, size, fp)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: suspicious comparison against 0; 'fread' returns an unsigned 'size_t', so comparing it with '0 >=' is equivalent to comparing it with '== 0'. To detect short reads or writes, compare against the 'nmemb' argument
    // CHECK-FIXES: if (size != fread(buf, 1, size, fp)) {
  }

  if (0 < fread(buf, 1, size, fp)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: return value of 'fread' is compared to 0; since 'nmemb' is not 1, partial reads or writes cannot be handled. Compare against the 'nmemb' argument instead
    // CHECK-FIXES: if (size != fread(buf, 1, size, fp)) {
  }
}

void test_discarded_result(FILE *fp, void *buf, size_t size) {
  if (fread(buf, 1, size, fp) == 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: return value of 'fread' is compared to 0; since 'nmemb' is not 1, partial reads or writes cannot be handled. Compare against the 'nmemb' argument instead
    // CHECK-FIXES: if (fread(buf, 1, size, fp) != size) {
  }

  if (fread(buf, 1, size, fp) != 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: return value of 'fread' is compared to 0; since 'nmemb' is not 1, partial reads or writes cannot be handled. Compare against the 'nmemb' argument instead
    // CHECK-FIXES: if (fread(buf, 1, size, fp) != size) {
  }

  if (fread(buf, 1, size, fp) > 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: return value of 'fread' is compared to 0; since 'nmemb' is not 1, partial reads or writes cannot be handled. Compare against the 'nmemb' argument instead
    // CHECK-FIXES: if (fread(buf, 1, size, fp) != size) {
  }

  // If nmemb is exactly 1, == 0 is safe because it's an all-or-nothing read
  if (fread(buf, size, 1, fp) == 0) {
    // No warning
  }
  
  // Streaming/EOF checks (like while(> 0)) are valid if not a direct comparison (e.g., they assign or loop).
  // However, this check only catches direct comparisons to avoid breaking legitimate streaming loops.
  while (fwrite(buf, 1, size, fp) > 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: return value of 'fwrite' is compared to 0; since 'nmemb' is not 1, partial reads or writes cannot be handled. Compare against the 'nmemb' argument instead
    // CHECK-FIXES: while (fwrite(buf, 1, size, fp) != size) {
  }
}

void test_correct_handling(FILE *fp, void *buf, size_t size) {
  if (fwrite(buf, 1, size, fp) != size) {
    // No warning
  }
  
  if (fwrite(buf, 1, size, fp) < size) {
    // No warning
  }

  size_t written = fwrite(buf, 1, size, fp);
  if (written < size) {
    // No warning
  }
  
  if (written <= 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: suspicious comparison against 0; 'fwrite' returns an unsigned 'size_t', so comparing it with '<= 0' is equivalent to comparing it with '== 0'. To detect short reads or writes, compare against the 'nmemb' argument
    // CHECK-FIXES: if (written != size) {
  }
}

#define SIZE 16

void test_macro_handling(FILE *fp, void *buf) {
  if (fwrite(buf, 1, SIZE, fp) <= 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: suspicious comparison against 0; 'fwrite' returns an unsigned 'size_t', so comparing it with '<= 0' is equivalent to comparing it with '== 0'. To detect short reads or writes, compare against the 'nmemb' argument
    // CHECK-FIXES: if (fwrite(buf, 1, SIZE, fp) != SIZE) {
  }
}
