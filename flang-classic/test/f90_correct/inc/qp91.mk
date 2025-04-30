#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtoslog  ########


qp91: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp91.f08 fcheck.$(OBJX)
	-$(RM) qp91.$(EXESUFFIX) qp91.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp91.f08 -o qp91.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp91.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp91.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp91
	qp91.$(EXESUFFIX)

verify: ;

qp91.run: run

