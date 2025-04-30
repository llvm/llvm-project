#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


qc17: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qc17.f08 fcheck.$(OBJX)
	-$(RM) qc17.$(EXESUFFIX) qc17.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qc17.f08 -o qc17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qc17.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qc17.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qc17
	qc17.$(EXESUFFIX)

verify: ;

qc17.run: run

