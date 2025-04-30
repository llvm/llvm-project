#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test fold xtoi  ########


qp99: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp99.f08 fcheck.$(OBJX)
	-$(RM) qp99.$(EXESUFFIX) qp99.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp99.f08 -o qp99.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp99.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp99.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp99
	qp99.$(EXESUFFIX)

verify: ;

qp99.run: run

