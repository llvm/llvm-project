#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtoblog  ########


qp112: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp112.f08 fcheck.$(OBJX)
	-$(RM) qp112.$(EXESUFFIX) qp112.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp112.f08 -o qp112.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp112.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp112.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp112
	qp112.$(EXESUFFIX)

verify: ;

qp112.run: run

