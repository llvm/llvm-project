# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

build:
	@echo ------------------------------------- building test $@
	$(FC) $(FFLAGS) $(SRC)/$(TEST).f90 -o $(TEST).$(EXE)
	 
run:
	@echo ------------------------------------ executing test $@
	-$(CP) $(SRC)/nl5.dat .
	./$(TEST).$(EXE)
	 
verify: ;
