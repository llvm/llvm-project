TEST(VecLibBitfieldTest, EncodingDecodingTest) {
  clang::CodeGenOptions Opts;

  // Test encoding and decoding for each vector library
  for (int i = static_cast<int>(llvm::driver::VectorLibrary::Accelerate);
       i <= static_cast<int>(llvm::driver::VectorLibrary::MaxLibrary); ++i) {

    Opts.VecLib = static_cast<llvm::driver::VectorLibrary>(i);

    // Encode and then decode
    llvm::driver::VectorLibrary decodedValue =
        static_cast<llvm::driver::VectorLibrary>(Opts.VecLib);

    EXPECT_EQ(decodedValue, Opts.VecLib)
        << "Encoding/Decoding failed for vector library " << i;
  }
}
