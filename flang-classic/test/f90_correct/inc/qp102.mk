#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


qp102: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp102.f08 fcheck.$(OBJX)
	-$(RM) qp102.$(EXESUFFIX) qp102.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp102.f08 -o qp102.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp102.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp102.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp102
	qp102.$(EXESUFFIX)

verify: ;

qp102.run: run

