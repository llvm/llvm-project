extern "C" int x;

namespace {

struct Init {
public:
  Init() { x = 1; }
};

Init SetX;

} // namespace
