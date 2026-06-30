module {
  func.func @f2(%arg0: vector<16xi16>) -> vector<3xi16> {
    %0 = arith.trunci %arg0 : vector<16xi16> to vector<16xi3>
    %1 = vector.bitcast %0 : vector<16xi3> to vector<3xi16>
    return %1 : vector<3xi16>
  }
}
