// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// pr19682

@interface Subscriptable
- (id)objectForKeyedSubscript:(id)sub __attribute__((unavailable)); // expected-note 2 {{'objectForKeyedSubscript:' has been explicitly marked unavailable here}}
- (void)setObject:(id)object forKeyedSubscript:(id)key __attribute__((unavailable)); // expected-note {{'setObject:forKeyedSubscript:' has been explicitly marked unavailable here}}
@end

id test(Subscriptable *obj) {
  obj[obj] = obj;  // expected-error {{'setObject:forKeyedSubscript:' is unavailable}}
  return obj[obj]; // expected-error {{'objectForKeyedSubscript:' is unavailable}}
}

id control(Subscriptable *obj) {
  return [obj objectForKeyedSubscript:obj]; // expected-error {{'objectForKeyedSubscript:' is unavailable}}
}

