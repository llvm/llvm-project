#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtosint  ########


qp90: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp90.f08 fcheck.$(OBJX)
	-$(RM) qp90.$(EXESUFFIX) qp90.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp90.f08 -o qp90.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp90.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp90.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp90
	qp90.$(EXESUFFIX)

verify: ;

qp90.run: run

