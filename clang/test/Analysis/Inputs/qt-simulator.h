#pragma clang system_header

namespace QtPrivate {
struct QSlotObjectBase {};
}

namespace Qt {
enum ConnectionType {};
enum TimerType {};
}

struct QMetaObject {
  struct Connection {};
};

struct QObject {
  static QMetaObject::Connection connectImpl(const QObject *, void **,
                                             const QObject *, void **,
                                             QtPrivate::QSlotObjectBase *,
                                             Qt::ConnectionType,
                                             const int *, const QMetaObject *);
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

struct QTimer {
  static void singleShotImpl(int, Qt::TimerType, const QObject *,
                             QtPrivate::QSlotObjectBase *);
};
