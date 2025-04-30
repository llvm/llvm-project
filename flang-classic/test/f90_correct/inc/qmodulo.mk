#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qmodulo function take quadruple precision  ########


qmodulo: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qmodulo.f08 fcheck.$(OBJX)
	-$(RM) qmodulo.$(EXESUFFIX) qmodulo.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qmodulo.f08 -o qmodulo.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qmodulo.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qmodulo.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qmodulo
	qmodulo.$(EXESUFFIX)

verify: ;

qmodulo.run: run

