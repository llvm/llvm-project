#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
SRC2=$(SRC)/src
build:
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC2)/$(TEST).f90 -o $(TEST).$(OBJX)
	-$(FC) $(LDFLAGS) $(TEST).$(OBJX) -o $(TEST).$(EXE)
run: fs25864.$(OBJX)
	$(RUN4) $(TEST).$(EXE)
