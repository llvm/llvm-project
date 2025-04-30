#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


qp50: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp50.f08 fcheck.$(OBJX)
	-$(RM) qp50.$(EXESUFFIX) qp50.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp50.f08 -o qp50.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp50.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp50.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp50
	qp50.$(EXESUFFIX)

verify: ;

qp50.run: run

