using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLVM.ClangTidy
{
    /// <summary>
    /// Allows entire categories of properties to be enabled, disabled, or inherited
    /// in one fell swoop.  We add properties to each category with the value being
    /// this enum, and when the value is selected, we use reflection to find all other
    /// properties in the same category and perform the corresponding action.
    /// </summary>
    public enum CategoryVerb
    {
        None,
        Disable,
        Enable,
        Inherit
    }

    public class CategoryVerbConverter : EnumConverter
    {
        public CategoryVerbConverter() : base(typeof(CategoryVerb))
        {
        }

        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            if (value is string)
            {
                switch ((string)value)
                {
                    case "Disable Category":
                        return CategoryVerb.Disable;
                    case "Enable Category":
                        return CategoryVerb.Enable;
                    case "Inherit Category":
                        return CategoryVerb.Inherit;
                    case "":
                        return CategoryVerb.None;
                }
            }
            return base.ConvertFrom(context, culture, value);
        }

        public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
        {
            if (value is CategoryVerb && destinationType == typeof(string))
            {
                switch ((CategoryVerb)value)
                {
                    case CategoryVerb.Disable:
                        return "Disable Category";
                    case CategoryVerb.Enable:
                        return "Enable Category";
                    case CategoryVerb.Inherit:
                        return "Inherit Category";
                    case CategoryVerb.None:
                        return String.Empty;
                }
            }

            return base.ConvertTo(context, culture, value, destinationType);
        }
    }
}
