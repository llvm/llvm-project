#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


qc18: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qc18.f08 fcheck.$(OBJX)
	-$(RM) qc18.$(EXESUFFIX) qc18.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qc18.f08 -o qc18.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qc18.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qc18.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qc18
	qc18.$(EXESUFFIX)

verify: ;

qc18.run: run

