STRING_EXTENSION_OUTSIDE(SBVariableAnnotator)

    % extend lldb::SBVariableAnnotator {
#ifdef SWIGPYTHON
  % pythoncode % {
        def get_annotations_list(self, instruction):
            """Get variable annotations as a Python list of dictionaries.

            Args:
                instruction: SBInstruction object to annotate

            Returns:
                List of dictionaries, each containing variable annotation data
            """
            structured_data = self.AnnotateStructured(instruction)
            if not structured_data.IsValid():
                return []

            annotations = []
            for i in range(structured_data.GetSize()):
                item = structured_data.GetItemAtIndex(i)
                if item.GetType() != lldb.eStructuredDataTypeDictionary:
                    continue

                annotation = {}

                boolean_fields = ['is_live']
                integer_fields = ['start_address', 'end_address', 'register_kind', 'decl_line']
                string_fields = ['variable_name', 'location_description', 'decl_file', 'type_name']

                for field in boolean_fields:
                    value = item.GetValueForKey(field)
                    if value.IsValid():
                        annotation[field] = value.GetBooleanValue()

                for field in integer_fields:
                    value = item.GetValueForKey(field)
                    if value.IsValid():
                        annotation[field] = value.GetUnsignedIntegerValue()

                for field in string_fields:
                    value = item.GetValueForKey(field)
                    if value.IsValid():
                        annotation[field] = value.GetStringValue(1024)

                annotations.append(annotation)

            return annotations
    %
  }
#endif
}