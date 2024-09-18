// Simulate the addition of a new library without increasing the bitfield size
enum class SimulatedVectorLibrary {
  Accelerate = 0,
  LIBMVEC,
  MASSV,
  SVML,
  SLEEF,
  Darwin_libsystem_m,
  ArmPL,
  AMDLIBM,
  NoLibrary,
  // Simulate new addition
  NewLibrary,
  MaxLibrary
};

#define SIMULATED_VECLIB_BIT_COUNT                                             \
  4 // The current bitfield size (should be 4 for 9 options)

TEST(VecLibBitfieldTest, SimulatedOverflowTest) {
  // Simulate the addition of a new library and check if the bitfield size is
  // sufficient
  EXPECT_LE(static_cast<size_t>(SimulatedVectorLibrary::MaxLibrary),
            (1 << SIMULATED_VECLIB_BIT_COUNT))
      << "Simulated VecLib bitfield size overflow!";
}
