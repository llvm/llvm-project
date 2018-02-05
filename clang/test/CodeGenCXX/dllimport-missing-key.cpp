// RUN: %clang_cc1 -triple i686-windows-gnu -emit-llvm -std=c++1y -O0 -o - %s -w | FileCheck --check-prefix=GNU %s

class __declspec(dllimport) QObjectData {
public:
    virtual ~QObjectData() = 0;
    void *ptr;

    int method() const;
};

class LocalClass : public QObjectData {
};

void call() {
    (new LocalClass())->method();
}

// GNU-DAG: @_ZTV11QObjectData = available_externally dllimport
// GNU-DAG: @_ZTS11QObjectData = linkonce_odr
// GNU-DAG: @_ZTI11QObjectData = linkonce_odr
