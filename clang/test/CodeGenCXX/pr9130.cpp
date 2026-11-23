// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s %}

class nsOggCodecState {
  virtual int StartTime() {
    return -1;
  }
};
class nsVorbisState : public nsOggCodecState {
  virtual ~nsVorbisState();
};
nsVorbisState::~nsVorbisState() {
}

// CHECK-LABEL: define linkonce_odr noundef i32 @_ZN15nsOggCodecState9StartTimeEv
