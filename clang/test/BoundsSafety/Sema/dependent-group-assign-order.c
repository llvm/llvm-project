
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct CountedByData {
  unsigned len;
  char *__counted_by(len) buf;
};

void TestOrderImplOK(struct CountedByData *data) {
  data->len -= 10;
  data->buf += 10;
}


void TestOrderImplOK2(struct CountedByData *data) {
  data->len = 4;
  data->buf = data->buf + 10;
}

void TestOrderOK(struct CountedByData *data) {
  data->buf += 10;
  data->len -= 10;
}

void TestOrderRefOK(struct CountedByData *data) {
  data->buf = data->buf + 1;
  data->len -= 1;
}

void TestOutParamOrderImplOK(char *__sized_by(*outLen) *outBuf, unsigned *outLen) {
  *outLen = 10;
  *outBuf += 1;
}

void *__sized_by(l) alloc(unsigned long long l);

void TestOrderPtrPromoteFail1(struct CountedByData *data, unsigned new_len) {
  data->len = new_len;
  data->buf = alloc(new_len); // expected-error{{assignments to dependent variables should not have side effects between them}}
}

void TestOrderPtrPromoteFail2(struct CountedByData *data, unsigned new_len) {
  data->len = new_len; // expected-note{{'data->len' has been assigned here}}
  // expected-error@+1{{assignments to dependent variables should not have side effects between them}}
  data->buf = alloc(data->len); // expected-error{{cannot reference 'data->len' after it is changed during consecutive assignments}}
}

void TestOrderPtrPromoteOK(struct CountedByData *data, unsigned new_len) {
  data->buf = alloc(new_len);
  data->len = new_len;
}

unsigned glen;
char *__counted_by(glen) gbuf;

void TestOrderGlobalPtrPromoteFail1(unsigned new_len) {
  glen = new_len;
  gbuf = alloc(new_len); // expected-error{{assignments to dependent variables should not have side effects between them}}
}

void TestOrderGlobalPtrPromoteFail2(unsigned new_len) {
  glen = new_len; // expected-note{{'glen' has been assigned here}}
  // expected-error@+1{{assignments to dependent variables should not have side effects between them}}
  gbuf = alloc(glen); // expected-error{{cannot reference 'glen' after it is changed during consecutive assignments}}
}

void TestOrderGlobalPtrPromoteOK(unsigned new_len) {
  gbuf = alloc(new_len);
  glen = new_len;
}

void TestOrderParmPtrPromoteFail1(unsigned new_len,
                                  char *__counted_by(len) buf, unsigned len) {
  len = new_len;
  buf = alloc(new_len); // expected-error{{assignments to dependent variables should not have side effects between them}}
}

void TestOrderParmPtrPromoteFail2(unsigned new_len,
                                  char *__counted_by(len) buf, unsigned len) {
  len = new_len; // expected-note{{'len' has been assigned here}}
  // expected-error@+1{{assignments to dependent variables should not have side effects between them}}
  buf = alloc(len); // expected-error{{cannot reference 'len' after it is changed during consecutive assignments}}
}

void TestOrderParmPtrPromoteOK(unsigned new_len,
                               char *__counted_by(len) buf, unsigned len) {
  buf = alloc(new_len);
  len = new_len;
}

void TestOrderLocalPtrPromoteFail1(unsigned new_len) {
  unsigned len;
  char *__counted_by(len) buf;
  len = new_len;
  buf = alloc(new_len); // expected-error{{assignments to dependent variables should not have side effects between them}}
}

void TestOrderLocalPtrPromoteFail2(unsigned new_len) {
  unsigned len;
  char *__counted_by(len) buf;
  // expected-note@+1{{'len' has been assigned here}}
  len = new_len;
  // expected-error@+1{{assignments to dependent variables should not have side effects between them}}
  buf = alloc(len); // expected-error{{cannot reference 'len' after it is changed during consecutive assignments}}
}

void TestOrderLocalPtrPromoteOK(unsigned new_len) {
  unsigned len;
  char *__counted_by(len) buf;
  buf = alloc(new_len);
  len = new_len;
}


struct CountedByData2 {
  unsigned len;
  char *__counted_by(len) buf;
  char *__counted_by(len) buf2;
};

void TestOrderRefOK2(struct CountedByData2 *data) {
  data->buf2 = data->buf; // old data->buf with old data->len
  data->buf = data->buf + 1;
  data->len -= 1;
}

void TestOrderRefFail(struct CountedByData2 *data) {
  data->buf = data->buf + 1; // expected-note{{'data->buf' has been assigned here}}
  data->buf2 = data->buf; // expected-error{{cannot reference 'data->buf' after it is changed during consecutive assignments}}
  data->len -= 1;
}

void TestOrderRefFail2(struct CountedByData2 *data) {
  data->len = 10; // expected-note{{'data->len' has been assigned here}}
  data->buf = data->buf + data->len; // expected-error{{cannot reference 'data->len' after it is changed during consecutive assignments}}
  data->buf2 = data->buf2 + 1;
}

struct EndedByData {
  char *__ended_by(iter) start;
  char *__ended_by(end) iter;
  char *end;
};

void TestEndedByImplOK(struct EndedByData *data) {
  data->end -= 1;
  data->iter += 1;
  data->start = data->start;
}

void TestEndedByOK(struct EndedByData *data) {
  data->iter += 1;
  data->end -= 1;
  data->start = data->start;
}
