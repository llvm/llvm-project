#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qmin(default real 16)  ########


qmin: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qmin.f08 fcheck.$(OBJX)
	-$(RM) qmin.$(EXESUFFIX) qmin.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qmin.f08 -o qmin.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qmin.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qmin.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qmin
	qmin.$(EXESUFFIX)

verify: ;

qmin.run: run

