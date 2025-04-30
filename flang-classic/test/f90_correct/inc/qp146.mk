#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test fold_const  ########


qp146: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp146.f08 fcheck.$(OBJX)
	-$(RM) qp146.$(EXESUFFIX) qp146.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp146.f08 -o qp146.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp146.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp146.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp146
	qp146.$(EXESUFFIX)

verify: ;

qp146.run: run

