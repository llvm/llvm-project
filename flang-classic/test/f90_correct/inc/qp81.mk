#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtoblog ########


qp81: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp81.f08 fcheck.$(OBJX)
	-$(RM) qp81.$(EXESUFFIX) qp81.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp81.f08 -o qp81.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp81.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp81.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp81
	qp81.$(EXESUFFIX)

verify: ;

qp81.run: run

