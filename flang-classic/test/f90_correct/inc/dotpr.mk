#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test dot_product  ########


dotpr: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/dotpr.f08 fcheck.$(OBJX)
	-$(RM) dotpr.$(EXESUFFIX) dotpr.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dotpr.f08 -o dotpr.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dotpr.$(OBJX) fcheck.$(OBJX) $(LIBS) -o dotpr.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dotpr
	dotpr.$(EXESUFFIX)

verify: ;

dotpr.run: run

