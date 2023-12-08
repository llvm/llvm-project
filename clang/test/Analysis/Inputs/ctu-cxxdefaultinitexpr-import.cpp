namespace QHashPrivate {
template <typename> int b;
struct Data;
} // namespace QHashPrivate

struct QDomNodePrivate {};
template <typename = struct QString> struct QMultiHash {
  QHashPrivate::Data *d = nullptr;
};

struct QDomNamedNodeMapPrivate {
  QMultiHash<> map;
};
struct QDomElementPrivate : QDomNodePrivate {
  QDomElementPrivate();
  void importee();
  QMultiHash<> *m_attr = nullptr;
};
// --------- common part end ---------

QDomElementPrivate::QDomElementPrivate() : m_attr{new QMultiHash<>} {}
void QDomElementPrivate::importee() { (void)QMultiHash<>{}; }
struct foo {
  QDomElementPrivate m = {};
  static const int value = (QHashPrivate::b<foo>, 22);
};
