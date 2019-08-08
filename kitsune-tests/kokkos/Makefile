CLANGXX=$(KITSUNE_HOME)/bin/clang++
OPT=$(KITSUNE_HOME)/bin/opt 

KXXFLAGS=-fkokkos 
CXXFLAGS=-std=c++14 -fPIC -I$(KITSUNE_HOME)/include 

LDFLAGS= -L$(KITSUNE_HOME)/lib -lkokkos -ldl -lrt 

types=kitsune kokkos

sources=parallel_for_sum1 parallel_for_sum2 

targets=$(foreach f,${sources}, $(foreach t, ${types}, ${f}_${t}))

all: ${targets}

%_kitsune: %.cpp
	${CLANGXX} ${KXXFLAGS} ${CXXFLAGS} -S -emit-llvm -o $@.ll $<
	@echo -n "ir line count: " && /usr/bin/wc -l $@.ll | cut -d ' ' -f1
	${CLANGXX} ${KXXFLAGS} ${CXXFLAGS} -o $@ $< ${LDFLAGS}

%_kokkos: %.cpp
	${CLANGXX} ${CXXFLAGS} -S -emit-llvm -o $@.ll $< 
	@echo -n "ir line count:" && /usr/bin/wc -l $@.ll | cut -d ' ' -f1 
	${CLANGXX} ${CXXFLAGS} -o $@ $< ${LDFLAGS}

clean:
	rm -f ${targets} *.ll *~ 


