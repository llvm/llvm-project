LLVM_CONFIG=llvm-config
CXX=clang++ -std=c++17 -O0
CXXFLAGS= `$(LLVM_CONFIG) --cppflags` -g -fPIC -fno-rtti
LDFLAGS=`$(LLVM_CONFIG) --ldflags` -Wl,-znodelete

all: llvm_lva_pass.so bitcode_generation lvapass 

llvm_lva_pass.so: llvm_lva_pass.o 
	$(CXX) $(CXXFLAGS) -shared ./build/live_variable_analysis_main_pass.o -o ./build/live_variable_analysis_main_pass.so $(LDFLAGS)

llvm_lva_pass.o: live_variable_analysis_main_pass.cpp
	$(CXX) -c live_variable_analysis_main_pass.cpp -o ./build/live_variable_analysis_main_pass.o $(CXXFLAGS)

clean: 
	rm -rf ./output/* ./build/*  ./bin/* && cd ./bin && mkdir ./bitcodes

llvmir_generation: 
	$(CXX) -c -emit-llvm $(CXXFLAGS) ./tests/test_case_5.cpp -o ./bin/test_case_5.bc

bitcode_generation: llvmir_generation
	opt -instnamer -mem2reg ./bin/test_case_5.bc -S -o ./bin/bitcodes/test_case_5_mem2reg.bc

lvapass: 
	opt -load ./build/live_variable_analysis_main_pass.so -lva < ./bin/bitcodes/test_case_5_mem2reg.bc -f 2> ./output/lvay_pass_5.txt
