#pragma clang system_header

struct SysBase {
  virtual void shutdown() = 0;
  virtual ~SysBase() = default;
};

struct SysService : SysBase {
  void shutdown() override {}
  ~SysService() override { shutdown(); } // no-warning
};
