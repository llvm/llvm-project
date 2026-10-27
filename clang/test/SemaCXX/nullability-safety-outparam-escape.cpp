// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify=nonnull
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 %s -verify=nullable

struct V { int x; };
struct Res { bool isOk() const; };
Res getRef(int, V *&out);
bool getPtr(int, V **out);
bool getConstRef(int, const V *&out);
bool getNullablePtr(int, V *_Nullable *out);
void peekConstPtr(int, V *const *in);
void peekConstRef(int, V *const &in);
V *find(int);
V *_Nullable pickNullable(int);

int ref_out(int k) {
  V *out = nullptr;
  if (getRef(k, out).isOk())
    return out->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
  return 0;
}

int ptr_out(int k) {
  V *out = nullptr;
  if (getPtr(k, &out))
    return out->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
  return 0;
}

int const_pointee_ref_out(int k) {
  const V *out = nullptr;
  if (getConstRef(k, out))
    return out->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
  return 0;
}

int const_ptr_does_not_escape(int k) {
  V *p = nullptr;
  peekConstPtr(k, &p);
  return p->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int const_ref_does_not_escape(int k) {
  V *p = nullptr;
  peekConstRef(k, p);
  return p->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int narrowing_survives_escape(int k) {
  V *p = find(k);
  if (!p)
    return 0;
  getPtr(k, &p);
  return p->x;
}

int stale_guard_must_not_narrow(int k) {
  V *_Nullable p = pickNullable(k);
  bool ok = p != nullptr;
  getNullablePtr(k, &p);
  if (ok)
    return p->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
  return 0;
}

int escape_through_alias(int k) {
  V *p = nullptr;
  V **pp = &p;
  getPtr(k, pp);
  return p->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int alias_mapping_survives_call(int k) {
  V *_Nullable p = pickNullable(k);
  V *_Nullable *pp = &p;
  getNullablePtr(k, pp);
  if (!p)
    return 0;
  *pp = pickNullable(k);
  return p->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int accepted_loss_escape_after_null(int k) {
  V *out = nullptr;
  getPtr(k, &out);
  return out->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}
