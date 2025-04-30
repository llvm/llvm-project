#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test PARAMETER  ########


qp45_f: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp45_f.f08 fcheck.$(OBJX)
	-$(RM) qp45_f.$(EXESUFFIX) qp45_f.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp45_f.f08 -o qp45_f.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp45_f.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp45_f.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp45_f
	qp45_f.$(EXESUFFIX)

verify: ;

qp45_f.run: run

