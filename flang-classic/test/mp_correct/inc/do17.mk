#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do17: do17.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do17.$(OBJX): $(SRC)/do17.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do17.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do17.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do17
run: ;
