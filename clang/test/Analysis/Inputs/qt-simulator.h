#pragma clang system_header

struct QObject {
};

struct QEvent {
  enum Type { None };
  QEvent(Type) {}
};

struct QCoreApplication : public QObject {
  static void postEvent(QObject *receiver, QEvent *event);
  static QCoreApplication *instance();
};

struct QApplication : public QCoreApplication {};
