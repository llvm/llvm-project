#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
fork_omp: fork_omp.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
fork.$(OBJX): $(SRC)/fork.c
	-$(CC) $(CFLAGS) $(SRC)/fork.c
fork_omp.$(OBJX): $(SRC)/fork_omp.F90 fork.$(OBJX) check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/fork_omp.F90
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) fork_omp.$(OBJX) fork.$(OBJX) check.$(OBJX) $(LIBS) \
		-o a.$(EXESUFFIX)
build: fork_omp
run: ;
