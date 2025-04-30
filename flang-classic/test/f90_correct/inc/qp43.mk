#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test sind cosd tand  ########


qp43: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp43.f08 fcheck.$(OBJX)
	-$(RM) qp43.$(EXESUFFIX) qp43.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp43.f08 -o qp43.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp43.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp43.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp43
	qp43.$(EXESUFFIX)

verify: ;

qp43.run: run

